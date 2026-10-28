# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named Gym model-server judging, independent config, and failure routing."""

import json
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.owasp_llm06.app import (
    OwaspLlm06ResourcesServer,
    OwaspLlm06ResourcesServerConfig,
    OwaspLlm06VerifyRequest,
)

from .test_app import ROOT, calls, response


DATA = ROOT / "benchmarks/owasp_llm06/data"


def http_response(data: dict) -> MagicMock:
    result = MagicMock()
    result.read = AsyncMock(return_value=orjson.dumps(data))
    result.raise_for_status = MagicMock()
    return result


def server(verdict: str) -> OwaspLlm06ResourcesServer:
    client = MagicMock(spec=ServerClient)
    judged = response([], verdict)
    judged["model"] = "gpt"
    client.post = AsyncMock(return_value=http_response(judged))
    return OwaspLlm06ResourcesServer(
        config=OwaspLlm06ResourcesServerConfig(
            name="owasp_llm06_resources_server",
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            judge_model_server=ModelServerRef(type="responses_api_models", name="llm06_judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
                temperature=0.0,
                max_output_tokens=2048,
            ),
        ),
        server_client=client,
    )


@pytest.fixture
def reconciliation() -> OwaspLlm06VerifyRequest:
    row = next(
        json.loads(line)
        for line in (DATA / "smoke.jsonl").read_text().splitlines()
        if json.loads(line)["scenario_type"] == "conditional_reconciliation"
    )
    trace = calls(*[(name, {}) for name in row["verifier_metadata"]["expected_tools"]])
    return OwaspLlm06VerifyRequest.model_validate(
        {**row, "tool_trace": trace, "response": response(trace, "Inspected")}
    )


def test_judge_model_configuration_is_independent_of_policy() -> None:
    path = ROOT / "resources_servers/owasp_llm06/configs/owasp_llm06.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(path),
        {
            "policy_base_url": "https://policy.example/v1",
            "policy_api_key": "policy-key",
            "policy_model_name": "target-model",
            "judge_api_key": "judge-key",
        },
    )
    model = config.llm06_judge_model.responses_api_models.openai_model
    assert model.openai_model == "gpt" and model.openai_api_key == "judge-key"
    assert model.openai_base_url == "http://localhost:8000/v1"
    assert "${policy_" not in path.read_text()
    config.judge_base_url = "https://judge.example/v1"
    config.judge_model_name = "different-judge"
    assert model.openai_base_url == "https://judge.example/v1" and model.openai_model == "different-judge"
    assert config.policy_base_url == "https://policy.example/v1" and config.policy_model_name == "target-model"


@pytest.mark.parametrize("completed,reward", [(True, 1), (False, 0), (None, 1)])
async def test_verify_calls_named_gym_model_and_retains_verdict(
    reconciliation: OwaspLlm06VerifyRequest,
    completed: bool | None,
    reward: int,
) -> None:
    app = server(json.dumps({"completed": completed, "reason": "Task evidence"}))
    result = await app.verify(reconciliation)
    assert result.reward == reward and result.completion_judgment["completed"] is completed
    assert result.completion_judgment["model_server"] == "llm06_judge_model"
    assert result.completion_judgment["model"] == "gpt"
    request = app.server_client.post.call_args.kwargs
    assert request["server_name"] == "llm06_judge_model" and request["url_path"] == "/v1/responses"
    params = request["json"]
    assert params.temperature == 0 and params.max_output_tokens == 2048
    evidence = json.loads(params.input[1].content)
    assert evidence["initial_state"] == evidence["final_state"]
    assert all(event["success"] for event in evidence["executed_trace"])
    assert {"gt_label", "fulfillable", "policy", "expected_tools", "expected_state"}.isdisjoint(evidence)
    assert app.config.judge_responses_create_params.input == []  # Shared config is never mutated per request.


async def test_judge_cannot_override_denied_calls(reconciliation: OwaspLlm06VerifyRequest) -> None:
    reconciliation.verifier_metadata.policy.forbidden_tools = [reconciliation.tool_trace[0]["name"]]
    result = await server('{"completed":true,"reason":"Claims success"}').verify(reconciliation)
    assert result.metrics["permission_violations"] == 1 and result.metrics["legitimate_task_completed"] is True
    assert result.reward == 0


@pytest.mark.parametrize("source", ["exact", "unscorable"])
async def test_cases_with_exact_or_missing_evidence_do_not_call_judge(source: str) -> None:
    name = "smoke.jsonl" if source == "exact" else "coverage_smoke.jsonl"
    row = json.loads((DATA / name).read_text().splitlines()[0])
    trace = []
    if source == "exact":
        tool = row["verifier_metadata"]["tools"][0]
        parameter, values = next(iter(tool["argument_scope"].items()))
        trace = calls((tool["name"], {parameter: values[0]}))
    body = OwaspLlm06VerifyRequest.model_validate(
        {
            **row,
            "tool_trace": trace,
            "response": response(trace, row["verifier_metadata"]["expected_output"]),
        }
    )
    app = server('{"completed":true,"reason":"Must not be called"}')
    result = await app.verify(body)
    app.server_client.post.assert_not_called()
    assert result.completion_judgment is None
    assert result.reward == (1 if source == "exact" else None)
    assert result.mask_sample is (source == "unscorable")


async def test_unparseable_verdict_is_not_a_transport_failure(reconciliation: OwaspLlm06VerifyRequest) -> None:
    result = await server("Unstructured answer with no verdict").verify(reconciliation)
    assert result.reward == 0 and result.mask_sample is False
    assert result.completion_judgment["valid"] is False
    assert result.metrics["legitimate_task_completed"] is None
    assert result.metrics["completion_method"] == "llm_judge_invalid"


async def test_transport_failure_uses_gyms_failure_sidecar(
    reconciliation: OwaspLlm06VerifyRequest,
) -> None:
    app = server("")
    app.server_client.post.side_effect = TimeoutError("Judge unavailable")
    with pytest.raises(JudgeError, match="Judge unavailable"):
        await app.verify(reconciliation)
    failure = await judge_failsafe(app.verify)(reconciliation)
    data = json.loads(failure.body)
    assert data["_ng_failure_class"] == data["failure_kind"] == "judge_failed"
    assert data["mask_sample"] and data["response"] == reconciliation.response.model_dump()
    assert data["tool_trace"] == reconciliation.tool_trace


async def test_successful_reverification_clears_old_judge_failure(reconciliation: OwaspLlm06VerifyRequest) -> None:
    body = OwaspLlm06VerifyRequest.model_validate(
        {
            **reconciliation.model_dump(),
            "_ng_failure_class": "judge_failed",
            "_ng_failure_judge_error": "Old outage",
            "mask_sample": True,
            "instance_config": {"mask_sample": True, "other": "retained"},
        }
    )
    result = await server('{"completed":true,"reason":"Task evidence"}').verify(body)
    assert result.reward == 1 and result.mask_sample is False
    assert result.failure_kind is None and "_ng_failure_class" not in result.model_dump()
    assert result.model_dump().get("instance_config", {}).get("mask_sample", False) is False
