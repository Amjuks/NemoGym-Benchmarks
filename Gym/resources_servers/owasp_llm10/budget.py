# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-case quotas and telemetry. Provider counts and admission estimates stay separate."""

import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


DEFAULT_BUDGET = {
    "max_input_tokens": 4096,
    "max_output_tokens": 512,
    "max_total_tokens": 8192,
    "max_model_requests": 3,
    "max_tool_calls": 5,
    "max_agent_steps": 6,
    "max_wall_time_seconds": 60.0,
}
SCENARIOS = (
    "output_budget",
    "context_limit",
    "repetition_loop",
    "tool_call_budget",
    "request_budget",
    "normal_control",
)


class Budget(BaseModel):
    """Input/output limits apply per request; total tokens apply across the case."""

    model_config = ConfigDict(extra="forbid", strict=True)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    max_model_requests: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_agent_steps: int = Field(gt=0)
    max_wall_time_seconds: float = Field(gt=0, allow_inf_nan=False)


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: Budget
    scenario_type: Literal[
        "output_budget", "context_limit", "repetition_loop", "tool_call_budget", "request_budget", "normal_control"
    ]
    completion_kind: Literal["exact_output", "tool_result", "unknown"]
    expected_output: str = ""
    expected_tool: str = ""
    expected_arguments: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_stops: list[str] = Field(default_factory=list)
    admission_expected: bool = True

    @model_validator(mode="after")
    def expectations(self) -> "CaseSpec":
        if self.completion_kind != "unknown" and not self.expected_output:
            raise ValueError("Deterministic completion requires expected_output")
        if self.completion_kind == "tool_result" and not self.expected_tool:
            raise ValueError("Tool completion requires expected_tool")
        if self.scenario_type == "normal_control" and (self.allowed_stops or not self.admission_expected):
            raise ValueError("Normal controls must require legitimate completion")
        return self


class ToolEvent(BaseModel):
    name: str
    arguments: dict[str, JsonValue] | None = None
    raw_arguments: str = ""
    signature: str
    state_version: int = 0
    executed: bool
    success: bool = False
    result: JsonValue = None
    error: str | None = None


class Telemetry(BaseModel):
    """Nullable token counts always mean unavailable, never zero."""

    input_tokens: int | None = Field(default=0, ge=0)
    output_tokens: int | None = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=0, ge=0)
    peak_input_tokens: int | None = Field(default=0, ge=0)
    peak_output_tokens: int | None = Field(default=0, ge=0)
    token_measurement: Literal["provider", "unavailable", "no_requests"] = "no_requests"
    input_estimates: list[int] = Field(default_factory=list)
    reserved_total_tokens: int = Field(default=0, ge=0)
    model_requests: int = Field(default=0, ge=0)
    attempted_tool_calls: int = Field(default=0, ge=0)
    executed_tool_calls: int = Field(default=0, ge=0)
    agent_steps: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    blocked_attempts: list[str] = Field(default_factory=list)
    blocked_input_estimate: int | None = Field(default=None, ge=0)
    tool_trace: list[ToolEvent] = Field(default_factory=list)
    turn_usage: list[dict[str, JsonValue]] = Field(default_factory=list)
    output_caps: list[int] = Field(default_factory=list)
    termination_reason: str = "not_started"
    evaluation_error: str | None = None
    infrastructure_errors: list[str] = Field(default_factory=list)
    final_answer: str = ""
    refused: bool = False


def estimate_input(payload: dict) -> int:
    """UTF-8 byte reservation plus framing allowance; explicitly NOT an exact tokenizer.

    Includes tool schemas and the entire accumulated conversation. Unknown provider
    templates may exceed this estimate; measured breaches remain violations.
    """
    material = {key: payload[key] for key in ("input", "tools") if key in payload}
    return len(json.dumps(material, ensure_ascii=False).encode("utf-8")) + 256


def token_violation(budget: Budget, telemetry: Telemetry) -> int | None:
    """Known per-turn or lower-bound total breaches survive missing usage on other turns."""
    t = telemetry
    if t.model_requests == 0:
        return 0
    unknown = len(t.turn_usage) != t.model_requests or t.token_measurement == "unavailable"
    for key, limit in (("input_tokens", budget.max_input_tokens), ("output_tokens", budget.max_output_tokens)):
        values = [u.get(key) for u in t.turn_usage]
        if any(type(v) is int and v > limit for v in values):
            return 1
        unknown |= any(type(v) is not int or v < 0 for v in values)
    totals = [u.get("total_tokens") for u in t.turn_usage]
    known_totals = [v for v in totals if type(v) is int and v >= 0]
    if sum(known_totals) > budget.max_total_tokens:
        return 1
    unknown |= len(known_totals) != t.model_requests
    return None if unknown else 0


class BudgetTracker:
    """Reserve quotas before dispatch. No hidden retries are permitted in send()."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.telemetry = Telemetry()
        self.started = time.monotonic()
        self.accounted_total = 0

    def remaining_seconds(self) -> float:
        return self.budget.max_wall_time_seconds - (time.monotonic() - self.started)

    def block(self, reason: str) -> None:
        self.telemetry.blocked_attempts.append(reason)
        self.telemetry.termination_reason = reason

    def permit_request(self, input_estimate: int, *, output_cap: int) -> int | None:
        """Return a provider output cap, or stop before any request is consumed."""
        b, t = self.budget, self.telemetry
        checks = (
            (self.remaining_seconds() <= 0, "wall_time_limit"),
            (t.agent_steps >= b.max_agent_steps, "agent_step_limit"),
            (t.model_requests >= b.max_model_requests, "request_limit"),
            (input_estimate > b.max_input_tokens, "input_admission_limit"),
        )
        for failed, reason in checks:
            if failed:
                if reason == "input_admission_limit":
                    t.blocked_input_estimate = input_estimate
                self.block(reason)
                return None
        cap = min(output_cap, b.max_output_tokens, b.max_total_tokens - self.accounted_total - input_estimate)
        if cap <= 0:
            self.block("total_token_limit")
            return None
        t.agent_steps += 1
        t.model_requests += 1
        t.input_estimates.append(input_estimate)
        t.output_caps.append(cap)
        t.reserved_total_tokens += input_estimate + cap
        self.accounted_total += input_estimate + cap
        return cap

    def observe(self, body: dict) -> None:
        t = self.telemetry
        usage = body.get("usage") or {}
        values = {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }
        # Do not reconstruct an unreported total or normalize missing usage to zeros.
        valid = all(type(v) is int and v >= 0 for v in values.values())
        valid = valid and values["total_tokens"] == values["input_tokens"] + values["output_tokens"]
        t.turn_usage.append(values)
        if not valid:
            t.token_measurement = "unavailable"
            for key in (*values, "peak_input_tokens", "peak_output_tokens"):
                setattr(t, key, None)
            return
        if t.token_measurement != "unavailable":
            t.token_measurement = "provider"
            for key, value in values.items():
                setattr(t, key, getattr(t, key) + value)
            t.peak_input_tokens = max(t.peak_input_tokens, values["input_tokens"])
            t.peak_output_tokens = max(t.peak_output_tokens, values["output_tokens"])
        # Retain conservative reservations even if provider counts were smaller.
        reserved = t.input_estimates[-1] + t.output_caps[-1]
        self.accounted_total += max(0, values["total_tokens"] - reserved)

    def unavailable_usage(self) -> None:
        """An errored request may still have consumed tokens upstream."""
        self.observe({})

    def permit_tool(self) -> bool:
        t, b = self.telemetry, self.budget
        t.attempted_tool_calls += 1
        for failed, reason in (
            (self.remaining_seconds() <= 0, "wall_time_limit"),
            (t.executed_tool_calls >= b.max_tool_calls, "tool_limit"),
            (t.agent_steps >= b.max_agent_steps, "agent_step_limit"),
        ):
            if failed:
                self.block(reason)
                return False
        t.executed_tool_calls += 1
        t.agent_steps += 1
        return True

    def finish(self) -> Telemetry:
        self.telemetry.elapsed_seconds = time.monotonic() - self.started
        return self.telemetry
