# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a model run using public configs and native benchmark dataset builders."""

import argparse
import json
import sys
from pathlib import Path

import yaml


def prepare_inputs(benchmark: str, destination: Path) -> int:
    """Adapt native prepared data to Gym's rollout input format; preserve grading fields."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if benchmark == "iheval":
        from benchmarks.iheval.prepare import _SRC_TEST, _ensure_source, convert_row

        _ensure_source()
        source = _SRC_TEST
        agent = "iheval_benchmark_simple_agent"
    else:
        from benchmarks.toolsandbox.prepare import OUTPUT_FPATH, prepare

        source = OUTPUT_FPATH if OUTPUT_FPATH.exists() else prepare()
        agent = "toolsandbox_benchmark_agent"
    count = 0
    with source.open() as inputs, destination.open("w") as output:
        for line in inputs:
            if not line.strip():
                continue
            row = json.loads(line)
            params = row.setdefault("responses_create_params", {"input": []})
            if benchmark == "iheval" and (
                any(message.get("role") == "tool" or message.get("tool_calls") for message in params["input"])
                or any("function" in tool for tool in params.get("tools", []))
            ):
                row = convert_row(row)
            row["agent_ref"] = {"type": "responses_api_agents", "name": agent}
            row["_ng_task_index"], row["_ng_rollout_index"] = count, 0
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    local = yaml.safe_load(Path("env.yaml").read_text()) if Path("env.yaml").exists() else {}
    settings = (local or {}).get("evaluation", {})
    parser.add_argument("--model", default=settings.get("model"))
    parser.add_argument(
        "--endpoint", default=settings.get("endpoint"), help="OpenAI-compatible base URL ending in /v1"
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh run directory")
    parser.add_argument(
        "--user-model", default=settings.get("user_model", "gpt-oss-20b"), help="ToolSandbox user-simulator alias"
    )
    parser.add_argument("--judge-model", default=settings.get("judge_model", "nemotron-3-content-safety"))
    parser.add_argument("--max-output-tokens", type=int, default=settings.get("max_output_tokens", 65536))
    parser.add_argument("--judge-max-output-tokens", type=int, default=settings.get("judge_max_output_tokens", 16384))
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/evaluation"), help="Configuration template directory"
    )
    args = parser.parse_args()
    if not args.model or not args.endpoint:
        parser.error("Set evaluation.model and evaluation.endpoint in env.yaml, or pass --model and --endpoint")
    if args.max_output_tokens <= 0 or args.judge_max_output_tokens <= 0:
        parser.error("Token budgets must be positive")
    if args.output.exists():
        parser.error(
            "Output already exists; choose a fresh directory. Resume existing runs with gym eval run --resume."
        )
    profile = args.profile.resolve()
    configs = {}
    benchmarks = [("iheval", "llm02_iheval", 18998), ("toolsandbox", "llm06_toolsandbox", 1032)]
    for name, folder, _ in benchmarks:
        path = profile / f"{name}.yaml"
        configs[name] = yaml.safe_load(path.read_text())
    configs["aegis"] = yaml.safe_load((profile / "aegis.yaml").read_text())
    output = args.output.resolve()
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    venvs = output / "venvs"
    components = {
        "resources_servers": ["iheval", "toolsandbox", "aegis_v4_safety"],
        "responses_api_models": ["inference_provider", "openai_model", "vllm_model"],
        "responses_api_agents": ["simple_agent", "toolsandbox_agent"],
        "environment_servers": ["legacy_agent"],
    }
    for kind, names in components.items():
        for name in names:
            component = venvs / kind / name
            component.mkdir(parents=True)
            (component / ".venv").symlink_to(root / ".venv", target_is_directory=True)
    for config in configs.values():
        config["uv_venv_dir"] = str(venvs)
    evaluations = []
    for name, folder, count in benchmarks:
        config = configs[name]
        config["split"] = "benchmark"
        config["policy_model_name"] = args.model
        config["policy_model"]["responses_api_models"]["inference_provider"].update(
            model=args.model, base_url=args.endpoint, api_key="${evaluation.api_key}"
        )
        config["responses_create_params"]["max_output_tokens"] = args.max_output_tokens
        if name == "toolsandbox":
            config["user_model"]["responses_api_models"]["openai_model"].update(
                openai_model=args.user_model, openai_base_url=args.endpoint, openai_api_key="${evaluation.api_key}"
            )
            config["toolsandbox_benchmark_resources_server"]["resources_servers"]["toolsandbox"].update(
                user_model_name=args.user_model, user_max_tokens=args.max_output_tokens
            )
        stage = output / folder / "rollouts/preprocessed_datasets"
        stage.mkdir(parents=True)
        prepared = stage / "benchmark.jsonl"
        actual_count = prepare_inputs(name, prepared)
        if actual_count != count:
            parser.error(f"{name}: expected {count} native inputs, found {actual_count}; no evaluation started")
        agent, impl = (
            ("iheval_benchmark_simple_agent", "simple_agent")
            if name == "iheval"
            else ("toolsandbox_benchmark_agent", "toolsandbox_agent")
        )
        config[agent]["responses_api_agents"][impl]["datasets"][0]["jsonl_fpath"] = str(prepared)
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        evaluations.append({"model": args.model, "benchmark": name, "expected_rows": count, "output": folder})
    configs["aegis"]["aegis_v4_model"]["responses_api_models"]["vllm_model"].update(
        model=args.judge_model, base_url=args.endpoint, api_key="${evaluation.api_key}"
    )
    configs["aegis"]["aegis_v4_safety"]["resources_servers"]["aegis_v4_safety"]["judge_responses_create_params"][
        "max_output_tokens"
    ] = args.judge_max_output_tokens
    (output / "aegis.yaml").write_text(yaml.safe_dump(configs["aegis"], sort_keys=False))
    (output / "llm02_iheval/aegis").mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "models": [args.model],
                "judge": args.judge_model,
                "user_simulator": args.user_model,
                "evaluations": evaluations,
            },
            indent=2,
        )
    )
    print(f"Prepared {output}. No evaluations were started.")


if __name__ == "__main__":
    main()
