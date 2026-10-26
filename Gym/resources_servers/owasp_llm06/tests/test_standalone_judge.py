# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone policy/tool/judge rollouts over independent HTTP endpoints, with no Gym services."""

import argparse
import json
from pathlib import Path

import pytest
from aiohttp import web
from omegaconf import OmegaConf

import nemo_gym.server_utils as server_utils
from aggregate import aggregate
from run_standalone import main

from .test_app import ROOT, response


@pytest.fixture
async def http_pool(monkeypatch: pytest.MonkeyPatch):
    # Each test owns its HTTP pool; prior runner tests use separate asyncio.run loops.
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    client = server_utils.set_global_aiohttp_client(
        server_utils.GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=16,
            global_aiohttp_connector_limit_per_host=16,
        )
    )
    yield client
    await client.close()


@pytest.mark.parametrize(
    "verdict,status,reward", [(True, 200, 1), (False, 200, 0), ("invalid", 200, 0), (None, 503, None)]
)
async def test_standalone_run_uses_independent_judge_and_preserves_failures(
    tmp_path: Path,
    http_pool: object,
    verdict: bool | str | None,
    status: int,
    reward: int | None,
) -> None:
    rows = [json.loads(line) for line in (ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["scenario_type"] == "conditional_reconciliation")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    requests = []

    async def policy(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(("policy", payload))
        assert request.headers["Authorization"] == "Bearer policy-key" and payload["model"] == "target"
        if payload["tools"][0]["name"] == "agency_probe":
            body = (
                response([{"name": "agency_probe", "arguments": '{"marker":"ok"}'}])
                if payload["tool_choice"] != "none"
                else response([], "ok")
            )
        elif not any(item.get("type") == "function_call_output" for item in payload["input"]):
            body = response([{"name": name, "arguments": "{}"} for name in row["verifier_metadata"]["expected_tools"]])
            body["output"] = [item for item in body["output"] if item["type"] == "function_call"]
        else:
            body = response([], "Inspected source systems")
        return web.json_response(body)

    async def judge(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(("judge", payload))
        assert request.headers["Authorization"] == "Bearer judge-key" and payload["model"] == "gpt"
        assert payload["temperature"] == 0 and payload["max_output_tokens"] == 2048
        assert "tools" not in payload
        evidence = json.loads(payload["input"][1]["content"])
        assert all(item["success"] for item in evidence["executed_trace"])
        assert evidence["assistant_answer"] == "Inspected source systems"
        assert {"gt_label", "fulfillable", "expected_tools", "policy"}.isdisjoint(evidence)
        if status != 200:
            return web.json_response({"error": {"message": "Judge unavailable"}}, status=status)
        text = (
            "No JSON verdict"
            if verdict == "invalid"
            else json.dumps({"completed": verdict, "reason": "Evidence checked"})
        )
        body = response([], text)
        body["model"] = "gpt"
        # Reproduce the real endpoint's nonstandard /responses usage envelope.
        body["usage"] = {"prompt_tokens": 41, "completion_tokens": 12, "total_tokens": 53, "reasoning_tokens": 0}
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/policy/v1/responses", policy)
    app.router.add_post("/judge/v1/responses", judge)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    origin = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    config = OmegaConf.load(ROOT / "resources_servers/owasp_llm06/configs/owasp_llm06.yaml")
    config.judge_base_url, config.judge_api_key = origin + "/judge/v1", "judge-key"
    model = config.llm06_judge_model.responses_api_models.openai_model
    model.max_http_attempts, model.max_rate_limit_wait_seconds = 1, 0
    config_path = tmp_path / "judge.yaml"
    OmegaConf.save(config, config_path)
    args = argparse.Namespace(
        risk="llm06",
        input=str(dataset),
        output=str(tmp_path / "shards/shard-0.jsonl"),
        base_url=origin + "/policy/v1",
        model="target",
        api_key="policy-key",
        shard_index=0,
        shard_count=1,
        limit=0,
        concurrency=1,
        timeout=10,
        retries=0,
        fsync_every=1,
        resume=True,
        max_output_tokens=256,
        max_turns=8,
        max_tool_calls=16,
        probe_only=False,
        judge_config=str(config_path),
        judge_env_file=None,
    )
    try:
        await main(args)
        record = json.loads(Path(args.output).read_text())
        assert record["reward"] == reward and record["model_response"] == "Inspected source systems"
        assert record["tool_trace"] and record["response"]["output"]
        assert len([item for name, item in requests if name == "judge"]) == 1
        if status == 200:
            assert record["error"] is None
            assert record["completion_judgment"]["usage"]["input_tokens"] == 41
            assert record["completion_judgment"]["usage"]["output_tokens"] == 12
            if verdict == "invalid":
                assert record["metrics"]["completion_method"] == "llm_judge_invalid"
            requested = len(requests)
            await main(args)
            assert len(requests) == requested  # Completed judged cases never call policy or judge again.
            config.judge_model_name = "different-judge"
            OmegaConf.save(config, config_path)
            with pytest.raises(ValueError, match="different input/model/sharding"):
                await main(args)
            assert len(requests) == requested
        else:
            assert record["error_kind"] == "judge_error" and record["metrics"]["security_checks_passed"]
        report = aggregate(tmp_path, shards=1, risk="llm06", input_path=dataset)
        assert report["overall"]["errors"] == (0 if status == 200 else 1)
        assert report["overall"]["scored"] == (1 if status == 200 else 0)
        metadata = Path(args.output).with_suffix(".run.json").read_text()
        assert "judge-key" not in metadata and "policy-key" not in metadata
        other = {
            **record,
            "case_id": rows[-1]["case_id"],
            "run_parameters": {
                **record["run_parameters"],
                "judge_profile": {**record["run_parameters"]["judge_profile"], "model": "another-judge"},
            },
        }
        Path(args.output).write_text(json.dumps(record) + "\n" + json.dumps(other) + "\n")
        with pytest.raises(ValueError, match="different judge profiles"):
            aggregate(tmp_path, shards=1, risk="llm06", expected_count=2)
    finally:
        await runner.cleanup()
