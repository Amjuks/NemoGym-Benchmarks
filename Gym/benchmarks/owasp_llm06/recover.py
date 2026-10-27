# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover supported failures separately, reusing saved policy responses for judge errors."""

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from aggregate import aggregate
from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client
from resources_servers.owasp_llm06.app import OwaspLlm06VerifyRequest
from resources_servers.owasp_llm06.standalone import StandaloneVerifier
from run_standalone import process


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


async def recover_judge(row: dict, original: dict, verifier: StandaloneVerifier) -> dict:
    """Reverify the saved response and trace without policy inference."""
    result = copy.deepcopy(original)
    request = OwaspLlm06VerifyRequest.model_validate(
        {**row, "response": original["response"], "tool_trace": original["tool_trace"]}
    )
    try:
        verdict = await verifier.verify(request)
    except JudgeError as exc:
        result.update(error=str(exc), error_kind="judge_error", reward=None)
    else:
        result.update(
            metrics=verdict.metrics,
            reward=verdict.reward,
            completion_judgment=verdict.completion_judgment,
            error=None,
            rollout_completed=True,
        )
        result.pop("error_kind", None)
    return result


async def main(args: argparse.Namespace) -> None:
    source, destination = args.source.resolve(), args.output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Recovery output must be separate from the source run")
    rows = read_rows(source / "dataset.jsonl")
    paths = sorted((source / "shards").glob("shard-*.jsonl"))
    originals = [r for path in paths for r in read_rows(path)]
    by_id = {r["case_id"]: r for r in originals}
    if len(by_id) != len(originals) or set(by_id) != {r["case_id"] for r in rows}:
        raise ValueError("Source run has duplicate/missing/unexpected cases")
    verifier = StandaloneVerifier.from_config(args.judge_config, env_file=args.env_file)
    private = OmegaConf.load(args.env_file)
    manifest = {
        "source": str(source),
        "source_shard_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "input_sha256": hashlib.sha256((source / "dataset.jsonl").read_bytes()).hexdigest(),
        "judge_profile": verifier.profile(),
        "retry_max_output_tokens": args.max_output_tokens,
        "retry_max_turns": args.max_turns,
        "note": "Original grading retained; per-case recovery metadata records retries and changed budgets.",
    }
    if destination.exists() and any(destination.iterdir()) and not (destination / "recovery.json").exists():
        raise ValueError("Output contains unrelated files; choose an empty recovery directory")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "recovery.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Recovery source/settings changed; use another output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("dataset.jsonl", "preparation_report.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    checkpoint = destination / "recovery_attempts.jsonl"
    recovered = {r["case_id"]: r for r in read_rows(checkpoint)} if checkpoint.exists() else {}
    pending = [
        r
        for r in rows
        if by_id[r["case_id"]].get("error")
        and r["verifier_metadata"]["scoring_available"]
        and (r["case_id"] not in recovered or args.retry_errors and recovered[r["case_id"]].get("error"))
    ]
    pending.sort(key=lambda row: by_id[row["case_id"]].get("error_kind") != "judge_error")
    if args.limit:
        pending = pending[: args.limit]
    implementation_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__),
            Path("resources_servers/owasp_llm06/standalone.py"),
            Path("resources_servers/owasp_llm06/mock_tools.py"),
        ]
    }
    pool = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    semaphore = asyncio.Semaphore(args.concurrency)

    async def attempt(row: dict) -> dict:
        original = by_id[row["case_id"]]
        latest = recovered.get(row["case_id"], original)
        if latest.get("error_kind") == "judge_error" and latest.get("response"):
            async with semaphore:
                result = await recover_judge(row, latest, verifier)
            mode = "saved_policy_response"
        else:
            settings = {
                **original["run_parameters"],
                "model": original["model"],
                "base_url": str(private.policy_base_url),
                "api_key": str(private.policy_api_key),
                "shard_index": original["shard_index"],
                "max_output_tokens": args.max_output_tokens,
                "max_turns": args.max_turns,
            }
            result = await process(row, verifier, argparse.Namespace(**settings), semaphore)
            mode = "policy_retry"
        result["recovery"] = {
            "mode": mode,
            "implementation_sha256": implementation_hashes,
            "original_error": original["error"],
            "judge_profile": verifier.profile(),
            "source": str(source),
        }
        return result

    try:
        with checkpoint.open("a", encoding="utf-8") as stream:
            for task in asyncio.as_completed([attempt(row) for row in pending]):
                result = await task
                recovered[result["case_id"]] = result
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {"case_id": result["case_id"], "error": result.get("error"), "reward": result.get("reward")}
                    ),
                    flush=True,
                )
    finally:
        await pool.close()
    shard_count = max(r["shard_index"] for r in originals) + 1
    (destination / "shards").mkdir(exist_ok=True)
    for index in range(shard_count):
        selected = [recovered.get(r["case_id"], r) for r in originals if r["shard_index"] == index]
        (destination / "shards" / f"shard-{index}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
        )
    report = aggregate(destination, shards=shard_count, risk="llm06", input_path=destination / "dataset.jsonl")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--judge-config", type=Path, default=Path("resources_servers/owasp_llm06/configs/owasp_llm06.yaml")
    )
    parser.add_argument("--env-file", type=Path, default=Path("env.yaml"))
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    options = parser.parse_args()
    if min(options.max_output_tokens, options.max_turns, options.concurrency) <= 0 or options.limit < 0:
        parser.error("Budgets/concurrency must be positive; limit must be nonnegative")
    asyncio.run(main(options))
