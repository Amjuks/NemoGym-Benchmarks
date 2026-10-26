# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web

from aggregate import aggregate
from benchmarks.owasp_llm06.prepare import augmented_spec
from benchmarks.owasp_llm06.reporting import build_summary, markdown
from resources_servers.owasp_llm06.agent import probe_tools
from run_standalone import load_rows, main

from .test_app import ROOT, response


def test_source_python_is_never_executed(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed"
    row = {
        "instruction": "First inspect, then if needed flag.",
        "python_codes": json.dumps([f"open({str(sentinel)!r}, 'w').write('bad')"]),
        "parameters": '[{"items": []}]',
        "tool_descs": '[[{"name":"run","parameters":{"type":"object"}}]]',
    }
    with pytest.raises(ValueError):
        augmented_spec(row)
    assert not sentinel.exists()


def test_probe_rejects_text_only_route() -> None:
    async def send(payload: dict) -> dict:
        return response([], "I can call tools")

    with pytest.raises(RuntimeError, match="forced function call"):
        asyncio.run(probe_tools(send, "text-only"))


def test_shared_runner_mock_backed_smoke_sharding_resume_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("No such file or directory: 'git'")

    monkeypatch.setattr("run_standalone.subprocess.run", missing_git)
    rows = [json.loads(s) for s in (ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().splitlines()]
    requests = []

    async def handler(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(payload)
        probe = payload["tools"][0]["name"] == "agency_probe"
        if probe:
            body = (
                response([{"name": "agency_probe", "arguments": '{"marker":"ok"}'}])
                if payload["tool_choice"] != "none"
                else response([], "ok")
            )
        else:
            instruction = payload["input"][1]["content"]
            row = next(r for r in rows if r["responses_create_params"]["input"][1]["content"] == instruction)
            spec = row["verifier_metadata"]
            completed = {i.get("name") for i in payload["input"] if i.get("type") == "function_call"}
            needed = [t for t in spec["tools"] if t["operation"] == "read" and t["name"] not in completed]
            needed += [
                t
                for t in spec["tools"]
                if t["operation"] != "read" and t["name"] in spec["expected_tools"] and t["name"] not in completed
            ]
            # The paired refill prompts are identical; use the visible tool result,
            # rather than selecting the outcome from the first matching prompt.
            if any(
                i.get("type") == "function_call_output" and "Needs doctor approval" in i.get("output", "")
                for i in payload["input"]
            ):
                needed = [t for t in needed if t["name"] != "request_medication_refill"]
            if needed:
                tool = needed[0]
                args = {k: v[0] for k, v in tool["argument_scope"].items()}
                args.update(tool["argument_subset"])
                body = response([{"name": tool["name"], "arguments": json.dumps(args)}])
                body["output"] = body["output"][:1]
                body["output"][0]["call_id"] = f"call_{len(completed)}"
                body["output"][0]["id"] = f"fc_{len(completed)}"
            else:
                body = response(
                    [],
                    spec.get("expected_output")
                    or "Inspection completed. Please obtain approval for any restricted action.",
                )
        return web.json_response(body)

    async def smoke() -> None:
        app = web.Application()
        app.router.add_post("/v1/responses", handler)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            for index in range(2):
                args = argparse.Namespace(
                    risk="llm06",
                    input=str(ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl"),
                    output=str(tmp_path / "shards" / f"shard-{index}.jsonl"),
                    base_url=f"http://127.0.0.1:{port}/v1",
                    model="mock-tool-model",
                    api_key="mock",
                    shard_index=index,
                    shard_count=2,
                    limit=0,
                    concurrency=2,
                    timeout=10,
                    retries=0,
                    fsync_every=1,
                    resume=True,
                    max_output_tokens=256,
                    max_turns=12,
                    max_tool_calls=32,
                    probe_only=False,
                )
                await main(args)
                requested = len(requests)
                await main(args)
                assert len(requests) == requested  # Resume never calls scored cases or re-probes.
            args.model = "different-model"
            with pytest.raises(ValueError, match="different input/model/sharding"):
                await main(args)
        finally:
            await server.cleanup()

    asyncio.run(smoke())
    report = aggregate(tmp_path, shards=2, risk="llm06", input_path=ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl")
    assert report["overall"]["total"] == 10 and report["overall"]["errors"] == 0
    assert report["overall"]["mean_reward"] == 1
    assert report["overall"]["legitimate_task_completion_rate"]["denominator"] == 8
    assert report["overall"]["legitimate_task_completion_rate"]["numerator"] == 5
    assert report["overall"]["approval_violation_rate"]["denominator"] == 4
    assert report["overall"]["approval_violation_rate"]["rate"] == 0
    assert (tmp_path / "dataset.jsonl").read_bytes() == (ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_bytes()
    assert (tmp_path / "report.md").is_file()
    for index in range(2):
        metadata = json.loads((tmp_path / "shards" / f"shard-{index}.run.json").read_text())
        assert metadata["git_commit"] is None
        assert len(metadata["input_sha256"]) == len(metadata["implementation_sha256"]) == 64
    assert len(load_rows(ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl", 0, 2, 0)) == 5
    with pytest.raises(ValueError, match="IDs differ"):
        aggregate(tmp_path, shards=2, risk="llm06", input_path=ROOT / "benchmarks/owasp_llm06/data/prepared.jsonl")


def test_errors_not_security_violations() -> None:
    records = [{"case_id": "failed", "model": "mock", "reward": None, "error": "HTTP timeout"}]
    summary = build_summary(records)
    assert summary["overall"]["errors"] == 1
    assert summary["overall"]["unauthorized_action_attempt_rate"]["rate"] is None
    assert "Unavailable" in markdown(summary, records)


def test_truncation_preserves_previous_tool_attempts() -> None:
    from resources_servers.owasp_llm06.agent import RolloutError, rollout

    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().splitlines()[0])
    tool = row["verifier_metadata"]["tools"][0]
    parameter, values = next(iter(tool["argument_scope"].items()))
    requests = 0

    async def send(payload: dict) -> dict:
        nonlocal requests
        requests += 1
        if requests == 1:
            body = response([{"name": tool["name"], "arguments": json.dumps({parameter: values[0]})}])
            body["output"] = body["output"][:1]
            return body
        raise TimeoutError("simulated gateway timeout")

    payload = {"model": "mock", "input": row["responses_create_params"]["input"]}
    with pytest.raises(RolloutError, match="TimeoutError") as error:
        asyncio.run(rollout(row, send, payload, max_turns=4, max_tool_calls=8))
    assert len(error.value.tool_trace) == 1
    assert error.value.tool_trace[0]["success"] is True


def test_probe_distinguishes_api_support_from_marker_acknowledgement() -> None:
    requests = []

    async def send(payload: dict) -> dict:
        requests.append(payload)
        if len(requests) == 1:
            return response([{"name": "agency_probe", "arguments": '{"marker":"ok"}'}])
        return response([], "Tool result received.")

    proof = asyncio.run(probe_tools(send, "mock"))
    assert proof["supported"] is True and proof["tool_result_round_trip"] is True
    assert proof["marker_acknowledged"] is False
    assert any(i.get("type") == "function_call_output" for i in requests[1]["input"])
