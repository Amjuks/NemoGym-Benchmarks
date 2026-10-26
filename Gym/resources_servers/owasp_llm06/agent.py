# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded Responses API tool loop, sharing the standalone runner's request transport."""

import json
from collections.abc import Awaitable, Callable

from resources_servers.owasp_llm06.mock_tools import CaseSpec, MockEnvironment


Send = Callable[[dict], Awaitable[dict]]


class RolloutError(RuntimeError):
    """An unscored rollout retaining all tool attempts before the failure."""

    def __init__(
        self,
        message: str,
        env: MockEnvironment,
        usages: list[dict],
        *,
        kind: str = "model_error",
        last_response: dict | None = None,
    ):
        super().__init__(message)
        self.tool_trace = env.trace
        self.turn_usage = usages
        self.kind = kind
        self.last_response = last_response


def tool_calls(body: dict) -> list[dict]:
    return [item for item in body.get("output", []) if item.get("type") == "function_call"]


async def probe_tools(send: Send, model: str) -> dict:
    """Require an actual function call and API acceptance of its tool-result input."""
    payload = {
        "model": model,
        "input": [
            {"role": "user", "content": "Call agency_probe with marker='ok' immediately; keep reasoning brief."}
        ],
        "tools": [
            {
                "type": "function",
                "name": "agency_probe",
                "description": "Tool calling capability probe",
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": "required",
        "max_output_tokens": 4096,
        "store": False,
        "stream": False,
    }
    body = await send(payload)
    calls = tool_calls(body)
    if len(calls) != 1 or calls[0].get("name") != "agency_probe" or not calls[0].get("call_id"):
        raise RuntimeError(
            f"Model route did not emit the forced function call (status={body.get('status')}); LLM06 evaluation aborted"
        )
    if json.loads(calls[0].get("arguments", "null")) != {"marker": "ok"}:
        raise RuntimeError("Model route emitted invalid capability probe arguments")
    payload["input"] += body["output"] + [
        {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": json.dumps({"marker": "ok"})}
    ]
    payload["tool_choice"] = "none"
    payload["input"].append({"role": "user", "content": "Repeat the marker returned by the tool."})
    followup = await send(payload)
    text = " ".join(
        p.get("text", "")
        for i in followup.get("output", [])
        for p in i.get("content", [])
        if p.get("type") == "output_text"
    )
    if not isinstance(followup.get("output"), list) or followup.get("status") in ("failed", "cancelled"):
        raise RuntimeError("Model route rejected the tool-result round trip; LLM06 evaluation aborted")
    return {
        "model": model,
        "supported": True,
        "api": "responses",
        "forced_call": calls[0],
        "tool_result_round_trip": True,
        "followup_status": followup.get("status"),
        "marker_acknowledged": "ok" in text.lower(),
    }


async def rollout(row: dict, send: Send, payload: dict, *, max_turns: int, max_tool_calls: int) -> dict:
    spec = CaseSpec.model_validate(row["verifier_metadata"])
    env = MockEnvironment(spec)
    inputs = list(payload["input"])
    outputs = []
    usages = []
    seen_calls = set()
    for _ in range(max_turns):
        try:
            tool_options = (
                {
                    "tools": [t.api_schema() for t in spec.tools],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
                if spec.tools
                else {}
            )
            body = await send(
                {
                    **payload,
                    "input": inputs,
                    **tool_options,
                }
            )
        except Exception as exc:
            raise RolloutError(
                f"Model request failed: {type(exc).__name__}: {exc}", env, usages, kind="api_error"
            ) from exc
        usages.append(body.get("usage") or {})
        if body.get("status") in ("failed", "cancelled", "incomplete"):
            raise RolloutError(
                f"Incomplete model turn: {body.get('status')}; details={body.get('incomplete_details')}",
                env,
                usages,
                kind="incomplete_generation",
                last_response=body,
            )
        if not isinstance(body.get("output"), list) or not body["output"]:
            raise RolloutError("Model API returned no trajectory output", env, usages)
        outputs.extend(body["output"])
        calls = tool_calls(body)
        if not calls:
            body["output"] = outputs
            return {"response": body, "tool_trace": env.trace, "turn_usage": usages, "stop_reason": "completed"}
        inputs.extend(body["output"])
        for call in calls:
            call_id = call.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                raise RolloutError("Missing or duplicate function call ID from model API", env, usages)
            seen_calls.add(call_id)
            if len(env.trace) >= max_tool_calls:
                raise RolloutError(
                    "Tool call limit reached; trajectory is unscored", env, usages, kind="trajectory_limit"
                )
            result = env.call(call.get("name", ""), call.get("arguments", ""))
            inputs.append(
                {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result, ensure_ascii=False)}
            )
    raise RolloutError("Turn limit reached; trajectory is unscored", env, usages, kind="trajectory_limit")
