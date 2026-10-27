# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery preserves policy evidence and distinguishes uncertain completion."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from benchmarks.owasp_llm06.recover import recover_judge
from benchmarks.owasp_llm06.reporting import stats
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.owasp_llm06.app import CompletionJudgeConfig, OwaspLlm06VerifyResponse
from resources_servers.owasp_llm06.standalone import JudgeEndpoint, StandaloneVerifier

from .test_app import ROOT, response


async def test_judge_http_body_preserved_and_secret_redacted(monkeypatch):
    verifier = StandaloneVerifier(
        CompletionJudgeConfig(),
        JudgeEndpoint(openai_base_url="http://localhost/v1", openai_api_key="private-key", openai_model="gpt"),
    )
    error = ClientResponseError(None, (), status=400, message="Bad Request")
    error.response_content = b'{"error":"invalid input private-key"}'
    monkeypatch.setattr(type(verifier.client), "create_response", AsyncMock(side_effect=error))
    with pytest.raises(JudgeError, match="HTTP 400:.*invalid input") as caught:
        await verifier.request_judge(NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert "private-key" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


async def test_recovery_uses_saved_response_and_does_not_mutate_original():
    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().split("\n")[0])
    original = {
        "response": response([], "saved answer"),
        "tool_trace": [],
        "reward": None,
        "error": "HTTP400",
        "error_kind": "judge_error",
    }
    before = copy.deepcopy(original)
    verifier = AsyncMock()
    verifier.verify.return_value = OwaspLlm06VerifyResponse(
        reward=1,
        metrics={"legitimate_task_completed": True},
        responses_create_params=row["responses_create_params"],
        response=original["response"],
    )
    result = await recover_judge(row, original, verifier)
    assert original == before
    assert result["response"] == before["response"] and result["tool_trace"] == []
    assert result["reward"] == 1 and result["error"] is None and "error_kind" not in result
    assert verifier.verify.call_args.args[0].response.output_text == "saved answer"
    verifier.verify.side_effect = JudgeError("provider explanation")
    failed = await recover_judge(row, original, verifier)
    assert failed["reward"] is None and failed["error"] == "provider explanation"
    assert original == before


def test_completion_evidence_metric_excludes_unknown_proxy():
    from resources_servers.owasp_llm06.mock_tools import CaseSpec, evaluate

    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().split("\n")[0])
    metrics = evaluate(CaseSpec.model_validate(row["verifier_metadata"]), [], answer="")
    records = [{"reward": 1, "metrics": {**metrics, "legitimate_task_completed": value}} for value in (True, None)]
    report = stats(records)
    assert report["mean_reward"] == 1
    assert report["completion_evidence_success_rate"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert report["workflow_only_positive_cases"] == 1


async def test_only_transient_header_rejection_is_retried(monkeypatch):
    verifier = StandaloneVerifier(
        CompletionJudgeConfig(),
        JudgeEndpoint(
            openai_base_url="http://localhost/v1", openai_api_key="test-key", openai_model="gpt", max_http_attempts=2
        ),
    )
    error = ClientResponseError(None, (), status=400, message="Bad Request")
    error.response_content = b'{"error":"could not decode header"}'
    call = AsyncMock(side_effect=[error, response([], '{"completed":true,"reason":"done"}')])
    monkeypatch.setattr(type(verifier.client), "create_response", call)
    monkeypatch.setattr("resources_servers.owasp_llm06.standalone.asyncio.sleep", AsyncMock())
    result = await verifier.request_judge(NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert result.output_text == '{"completed":true,"reason":"done"}'
    assert call.await_count == 2 and call.await_args_list[0] == call.await_args_list[1]
    call.reset_mock(side_effect=True)
    call.side_effect = error
    with pytest.raises(JudgeError, match="could not decode header"):
        await verifier.request_judge(NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert call.await_count == 2
    error.response_content = b'{"error":"invalid schema"}'
    call.reset_mock()
    with pytest.raises(JudgeError, match="invalid schema"):
        await verifier.request_judge(NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert call.await_count == 1


@pytest.mark.parametrize("operation", ["update", "read"])
def test_compound_source_guard_matches_all_fields(operation):
    import ast

    from benchmarks.owasp_llm06.adapters import compile_method
    from resources_servers.owasp_llm06.mock_tools import CaseSpec, MockEnvironment, Policy

    body = "record['quantity'] = quantity\n            return record" if operation == "update" else "return record"
    source = f"""def select(self, sale_id, item_id, quantity):
    for record in self.sales:
        if record.get('sale_id') == sale_id and record.get('item_id') == item_id:
            {body}
    return None
"""
    method = ast.parse(source).body[0]
    description = {
        "name": "select",
        "parameters": {
            "type": "object",
            "properties": {
                "sale_id": {"type": "string"},
                "item_id": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["sale_id", "item_id", "quantity"],
        },
    }
    records = [
        {"sale_id": "sale1", "item_id": "a", "quantity": 2},
        {"sale_id": "sale1", "item_id": "b", "quantity": 3},
    ]
    with pytest.raises(ValueError, match="unsupported_record_guard"):
        compile_method(method, description, {"sales": records}, {"sales": "sales"}, 0)
    tool = compile_method(method, description, {"sales": records}, {"sales": "sales"}, 0, compound_guards=True)
    spec = CaseSpec(
        tools=[tool],
        initial_state={"env0.sales": records},
        policy=Policy(allowed_tools=[tool.name], evidence=[source]),
        expected_tools=[tool.name],
        expected_behavior="Select matching sale and item",
    )
    env = MockEnvironment(spec)
    event = env.call(tool.name, json.dumps({"sale_id": "sale1", "item_id": "b", "quantity": 9}))
    assert event["success"]
    assert env.state["env0.sales"][0]["quantity"] == 2
    assert env.state["env0.sales"][1]["quantity"] == (9 if operation == "update" else 3)
    before = copy.deepcopy(env.state)
    missing = env.call(tool.name, json.dumps({"sale_id": "sale1", "item_id": "c", "quantity": 9}))
    assert missing["error"] == "unknown_record" and env.state == before
    method = ast.parse(source.replace(" and ", " or ")).body[0]
    with pytest.raises(ValueError, match="unsupported_record_guard"):
        compile_method(method, description, {"sales": records}, {"sales": "sales"}, 0, compound_guards=True)
