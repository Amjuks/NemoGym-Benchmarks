# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt bounded rollout telemetry to the installed Gym verifier request."""

from pydantic import ValidationError

from benchmarks.response_utils import normalize_usage
from resources_servers.owasp_llm10.agent import Send, rollout
from resources_servers.owasp_llm10.app import OwaspLlm10Verifier, OwaspLlm10VerifyRequest, evaluate
from resources_servers.owasp_llm10.budget import CaseSpec


async def execute(row: dict, send: Send, payload: dict, *, verifier: OwaspLlm10Verifier | None = None) -> dict:
    try:
        spec = CaseSpec.model_validate(row["verifier_metadata"])
    except (ValidationError, KeyError) as exc:
        return {
            "reward": None,
            "error": "invalid_case_configuration",
            "evaluation_error": str(exc),
            "metrics": {
                **{
                    key: None
                    for key in (
                        "reward",
                        "budget_compliant",
                        "termination_compliant",
                        "task_completed",
                        "token_budget_exceeded",
                        "request_budget_exceeded",
                        "tool_budget_exceeded",
                        "time_budget_exceeded",
                        "redundant_tool_calls",
                    )
                },
                "termination_reason": "invalid_case_configuration",
                "evaluation_error": "invalid_case_configuration",
            },
        }
    result = await rollout(row, send, payload)
    body = result["response"]
    typed_body = dict(body)
    # Normalization is ONLY for Gym envelope validation, after raw usage was recorded.
    # Unreported counts never enter the verifier through this response envelope.
    usage = body.get("usage") or {}
    if result["telemetry"]["token_measurement"] == "provider" and usage:
        typed_body = normalize_usage(typed_body)
    else:
        typed_body["usage"] = None
    try:
        request = OwaspLlm10VerifyRequest.model_validate({**row, **result, "response": typed_body})
        verdict = await (verifier or OwaspLlm10Verifier()).verify(request)
        metrics = verdict.metrics
    except ValidationError:
        # Retain the trajectory for diagnosis; never discard usage on schema failures.
        from resources_servers.owasp_llm10.budget import Telemetry

        result["telemetry"]["evaluation_error"] = "invalid_response_envelope"
        metrics = evaluate(spec, Telemetry.model_validate(result["telemetry"]))
    return {
        **result,
        "case_spec": spec.model_dump(),
        "metrics": metrics,
        "reward": metrics["reward"],
        "model_response": result["telemetry"]["final_answer"],
        "usage": {key: result["telemetry"][key] for key in ("input_tokens", "output_tokens", "total_tokens")},
        "error": metrics["evaluation_error"],
        "evaluation_error": metrics["evaluation_error"],
        "rollout_completed": not result["telemetry"]["infrastructure_errors"],
    }
