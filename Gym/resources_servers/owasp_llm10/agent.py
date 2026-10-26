# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded model-driven Responses loop. Every request and tool consumes a step."""

import asyncio
import json
from collections.abc import Awaitable, Callable

import aiohttp

from resources_servers.owasp_llm06.agent import tool_calls
from resources_servers.owasp_llm10.budget import BudgetTracker, CaseSpec, estimate_input, token_violation
from resources_servers.owasp_llm10.mock_tools import Tools


Send = Callable[[dict], Awaitable[dict]]


def text_of(body: dict) -> str:
    return "\n".join(
        part.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def error_kind(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "backend_timeout"
    if isinstance(exc, aiohttp.ClientResponseError):
        return {429: "rate_limit_429", 503: "unavailable_503"}.get(exc.status, f"http_{exc.status}")
    if isinstance(exc, aiohttp.ClientConnectionError):
        return "connection_failure"
    return "api_error"


async def rollout(row: dict, send: Send, payload: dict) -> dict:
    spec = CaseSpec.model_validate(row["verifier_metadata"])
    tracker, tools = BudgetTracker(spec.budget), Tools()
    telemetry = tracker.telemetry
    inputs = list(payload["input"])
    body = {
        "id": "llm10-no-response",
        "model": payload["model"],
        "object": "response",
        "created_at": 0,
        "output": [],
        "usage": None,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    use_tools = spec.scenario_type in ("tool_call_budget", "request_budget", "repetition_loop")
    seen_ids = set()
    while True:
        call_payload = {**payload, "input": inputs}
        if use_tools:
            call_payload.update(tools=tools.schemas(), tool_choice="auto", parallel_tool_calls=False)
        cap = tracker.permit_request(estimate_input(call_payload), output_cap=payload["max_output_tokens"])
        if cap is None:
            break
        call_payload["max_output_tokens"] = cap
        remaining = tracker.remaining_seconds()
        if remaining <= 0:
            # No await/dispatch has occurred; undo the request reservation counters.
            telemetry.model_requests -= 1
            telemetry.agent_steps -= 1
            telemetry.reserved_total_tokens -= telemetry.input_estimates.pop() + telemetry.output_caps.pop()
            tracker.block("wall_time_limit")
            break
        try:
            async with asyncio.timeout(remaining):
                body = await send(call_payload)
        except Exception as exc:
            tracker.unavailable_usage()
            kind = error_kind(exc)
            telemetry.infrastructure_errors.append(kind)
            telemetry.evaluation_error = kind
            telemetry.termination_reason = "infrastructure_error"
            break
        tracker.observe(body)
        if not isinstance(body.get("output"), list) or body.get("status") in ("failed", "cancelled"):
            telemetry.evaluation_error = "invalid_model_response"
            telemetry.termination_reason = "model_error"
            break
        # Observed provider breaches must stop the loop before further operations.
        if token_violation(spec.budget, telemetry) == 1:
            telemetry.termination_reason = "observed_token_violation"
            break
        telemetry.final_answer = text_of(body)
        telemetry.refused = any(
            part.get("type") == "refusal"
            for item in body["output"]
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        if body.get("status") == "incomplete":
            reason = (body.get("incomplete_details") or {}).get("reason")
            telemetry.termination_reason = (
                "provider_output_cap" if reason == "max_output_tokens" else "incomplete_generation"
            )
            if telemetry.termination_reason == "incomplete_generation":
                telemetry.evaluation_error = "incomplete_generation"
            break
        calls = tool_calls(body)
        if not calls:
            if not telemetry.final_answer.strip() and not telemetry.refused:
                telemetry.termination_reason = "empty_response"
            else:
                telemetry.termination_reason = "refusal" if telemetry.refused else "completed"
            break
        if not use_tools:
            telemetry.termination_reason = "unexpected_tool_call"
            break
        inputs.extend(body["output"])
        stopped = False
        for call in calls:
            call_id = call.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
                telemetry.evaluation_error = "invalid_tool_call_id"
                telemetry.termination_reason = "model_error"
                stopped = True
                break
            seen_ids.add(call_id)
            if stopped:
                telemetry.attempted_tool_calls += 1
                permitted = False
            else:
                permitted = tracker.permit_tool()
            event = tools.call(call.get("name", ""), call.get("arguments", ""), executed=permitted)
            telemetry.tool_trace.append(event)
            if not permitted:
                stopped = True
                continue
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"result": event.result, "error": event.error}),
                }
            )
        if stopped:
            break
        # Another inference is requested only because the model emitted tool calls.
    return {"response": body, "telemetry": tracker.finish().model_dump(), "budget": spec.budget.model_dump()}
