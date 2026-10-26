# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full source preservation, unknown scoring, and resume boundaries."""

import argparse
import asyncio
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from aggregate import aggregate
from benchmarks.owasp_llm06.prepare import REVISION, digest
from benchmarks.owasp_llm06.source_inventory import source_spec
from benchmarks.owasp_llm06.training_adapters import SchemaCatalog
from resources_servers.owasp_llm06.completion import apply_judgment
from resources_servers.owasp_llm06.mock_tools import CaseSpec, MockEnvironment, evaluate
from run_standalone import Gateway, main

from .test_app import ROOT, response


DATA = ROOT / "benchmarks/owasp_llm06/data"


def test_every_source_identity_and_payload_is_preserved() -> None:
    rows = [json.loads(line) for line in (DATA / "prepared.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["case_id"] for r in rows}) == 4220
    for split, count in (("train", 3464), ("augmented", 756)):
        selected = [r for r in rows if r["source_metadata"]["split"] == split]
        assert [r["source_metadata"]["row_index"] for r in selected] == list(range(count))
        for row in selected:
            source = row["source_metadata"]
            assert source["revision"] == REVISION and source["row_sha256"] == digest(row["source_row"])
            assert row["responses_create_params"]["input"][1]["content"] == row["source_row"]["instruction"]
            assert row["case_id"] == f"asb_{digest([REVISION, split, source['row_index'], source['row_sha256']])[:24]}"
    assert Counter(r["verifier_metadata"]["scoring_available"] for r in rows) == {True: 587, False: 3633}
    assert sum(r["duplicate_of"] is not None for r in rows) == 29
    assert not (DATA / "exclusions.jsonl").read_text()
    assert len((DATA / "unscored.jsonl").read_text().splitlines()) == 3633
    # Every former supported v2 source row still has its scoring adapter.
    supported = {
        (r["source_metadata"]["split"], r["source_metadata"]["row_index"])
        for r in rows
        if r["verifier_metadata"]["scoring_available"]
    }
    old = [json.loads(line) for line in (DATA / "v2/prepared.jsonl").read_text().splitlines()]
    assert supported == {(r["source_metadata"]["split"], r["source_metadata"]["row_index"]) for r in old}


def test_unknown_training_effects_are_errors_and_never_scores() -> None:
    row = {
        "id": 0,
        "environments": json.dumps(
            [
                {
                    "name": "Example",
                    "parameters": {"users": []},
                    "tools": ["create_user"],
                    "tool_desc": [{"name": "create_user", "parameters": {"type": "object", "properties": {}}}],
                }
            ]
        ),
    }
    spec, issues = source_spec(row, split="train", reason="unreviewed", catalog=SchemaCatalog([row]))
    assert not issues and spec.tools[0].operation == "unavailable"
    env = MockEnvironment(spec)
    before = copy.deepcopy(env.state)
    event = env.call(spec.tools[0].name, "{}")
    assert event["error"] == "unsupported_source_tool" and event["success"] is False
    assert env.state == before
    result = evaluate(spec, [dict(event, success=True)], answer="Done!")
    assert result["reward"] is None and result["unauthorized_action_attempt"] is None
    assert result["legitimate_task_completed"] is None
    assert apply_judgment(result, {"completed": True}) is None
    with pytest.raises(ValueError, match="completion expectations"):
        CaseSpec.model_validate({**spec.model_dump(), "scoring_available": True})


def test_invalid_source_code_is_retained_without_execution(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed"
    row = {
        "python_codes": json.dumps([f"open({str(sentinel)!r}, 'w').write('bad')\ninvalid syntax !"]),
        "parameters": '[{"items": []}]',
        "tool_descs": '[[{"name":"run","parameters":{"type":"object","properties":{}}}]]',
    }
    spec, issues = source_spec(row, split="augmented", reason="invalid_source_python", catalog=None)
    assert spec.tools[0].operation == "unavailable" and issues
    assert MockEnvironment(spec).call(spec.tools[0].name, "{}")["error"] == "unsupported_source_tool"
    assert not sentinel.exists()


def test_partial_augmented_adapter_keeps_working_reads() -> None:
    row = json.loads((DATA / "coverage_smoke.jsonl").read_text().splitlines()[1])
    spec = CaseSpec.model_validate(row["verifier_metadata"])
    tool = next(t for t in spec.tools if t.operation == "read" and not t.id_field)
    env = MockEnvironment(spec)
    event = env.call(tool.name, "{}")
    assert event["success"] and event["data"] == spec.initial_state[tool.collection]
    assert evaluate(spec, env.trace)["reward"] is None


def test_unscored_rollouts_resume_without_judging_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [json.loads(line) for line in (DATA / "coverage_smoke.jsonl").read_text().splitlines()]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []

    async def send(self: object, payload: dict) -> dict:
        calls.append(payload)
        if payload.get("tools") and payload["tools"][0]["name"] == "agency_probe":
            return (
                response([{"name": "agency_probe", "arguments": '{"marker":"ok"}'}])
                if payload["tool_choice"] != "none"
                else response([], "ok")
            )
        if payload["input"][1]["content"] == rows[2]["responses_create_params"]["input"][1]["content"]:
            assert "tools" not in payload and "tool_choice" not in payload and "parallel_tool_calls" not in payload
        return response([], "The available tools cannot complete this task.")

    def initialize(self: Gateway, args: argparse.Namespace) -> None:
        # The fake transport has no aiohttp pool or event-loop lifecycle.
        self.attempts = 0

    monkeypatch.setattr("run_standalone.Gateway.__init__", initialize)
    monkeypatch.setattr("run_standalone.Gateway.send", send)
    args = argparse.Namespace(
        risk="llm06",
        input=str(source),
        output=str(tmp_path / "shards/shard-0.jsonl"),
        base_url="http://unused/v1",
        model="mock",
        api_key="mock",
        shard_index=0,
        shard_count=1,
        limit=0,
        concurrency=2,
        timeout=10,
        retries=0,
        fsync_every=1,
        resume=True,
        max_output_tokens=256,
        max_turns=4,
        max_tool_calls=8,
        probe_only=False,
    )
    asyncio.run(main(args))
    requested = len(calls)
    asyncio.run(main(args))
    assert len(calls) == requested == 5  # Two probe calls and three actual rollouts.
    assert all(p["model"] != "must-not-be-called" for p in calls)
    report = aggregate(tmp_path, shards=1, risk="llm06", input_path=source)
    assert report["overall"]["total"] == report["overall"]["rollouts_completed"] == 3
    assert report["overall"]["scoring_unavailable"] == 3
    assert report["overall"]["scored"] == report["overall"]["errors"] == 0
    assert report["overall"]["unauthorized_action_attempt_rate"]["denominator"] == 0
    shard = Path(args.output)
    records = [json.loads(line) for line in shard.read_text().splitlines()]
    assert all(r["reward"] is None and r["metrics"]["evaluation_completed"] is False for r in records)
    records[0]["reward"] = 1
    shard.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="without a scoring adapter"):
        aggregate(tmp_path, shards=1, risk="llm06", input_path=source)
