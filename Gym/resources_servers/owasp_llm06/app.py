# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym trace verifier for the LLM06 sandbox."""

import asyncio
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.owasp_llm06.completion import (
    JUDGE_SYSTEM_MESSAGE,
    apply_judgment,
    completion_evidence,
    parse_judgment,
)
from resources_servers.owasp_llm06.mock_tools import CaseSpec, evaluate


class CompletionJudgeConfig(BaseModel):
    judge_model_server: ModelServerRef | None = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0, max_output_tokens=2048
        )
    )
    judge_system_message: str = JUDGE_SYSTEM_MESSAGE
    judge_endpoint_max_concurrency: int = Field(default=4, ge=1)


class OwaspLlm06ResourcesServerConfig(BaseResourcesServerConfig, CompletionJudgeConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class OwaspLlm06VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    case_id: str
    verifier_metadata: CaseSpec
    tool_trace: list[dict] = Field(default_factory=list)


class OwaspLlm06VerifyResponse(BaseVerifyResponse):
    reward: float | None
    metrics: dict
    completion_judgment: dict | None = None


class OwaspLlm06Verifier:
    async def judge_task(self, body: OwaspLlm06VerifyRequest) -> dict | None:
        """Deterministic-only callers have no configured judge dependency."""
        return None

    async def verify(self, body: OwaspLlm06VerifyRequest) -> OwaspLlm06VerifyResponse:
        calls = [item.model_dump() for item in body.response.output if item.type == "function_call"]
        if [(c.get("name"), c.get("arguments")) for c in calls] != [
            (c.get("name", ""), c.get("arguments", "")) for c in body.tool_trace
        ]:
            raise ValueError("Tool trace does not match model function calls")
        metrics = evaluate(body.verifier_metadata, body.tool_trace, answer=body.response.output_text or "")
        reward = metrics.pop("reward")
        judgment = None
        if body.verifier_metadata.scoring_available and metrics["legitimate_task_completed"] is None:
            judgment = await self.judge_task(body)
            if judgment is not None:
                reward = apply_judgment(metrics, judgment)
        fields = body.model_dump()
        # Reverification must replace previous verdicts rather than echo them as current evidence.
        fields.pop("_ng_failure_class", None)
        fields.pop("_ng_failure_judge_error", None)
        if isinstance(fields.get("instance_config"), dict) and "mask_sample" in fields["instance_config"]:
            fields["instance_config"]["mask_sample"] = reward is None
        fields.update(
            reward=reward,
            metrics=metrics,
            completion_judgment=judgment,
            mask_sample=reward is None,
            failure_kind="owasp_llm06:scoring_unavailable" if reward is None else None,
            failure_reason=body.verifier_metadata.scoring_unavailable_reason if reward is None else None,
        )
        return OwaspLlm06VerifyResponse(**fields)


class CompletionJudgeVerifier(OwaspLlm06Verifier):
    config: CompletionJudgeConfig
    _judge_semaphore: asyncio.Semaphore

    async def request_judge(self, params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError

    async def judge_task(self, body: OwaspLlm06VerifyRequest) -> dict | None:
        """Judge completion through the configured Gym model server, after deterministic replay."""
        reference = self.config.judge_model_server
        if reference is None:
            return None
        inputs = body.responses_create_params.input
        if isinstance(inputs, str):
            task = inputs
        else:
            task = next(
                (message.content for message in reversed(inputs) if getattr(message, "role", None) == "user"), ""
            )
            if not isinstance(task, str):
                task = json.dumps(task, default=lambda item: item.model_dump())
        evidence = completion_evidence(
            body.verifier_metadata, body.tool_trace, task=task, answer=body.response.output_text or ""
        )
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(role="system", content=self.config.judge_system_message),
            NeMoGymEasyInputMessage(role="user", content=json.dumps(evidence, ensure_ascii=False)),
        ]
        async with self._judge_semaphore:
            response = await self.request_judge(params)
        # Transport failures are JudgeError and Gym routes them to the failure sidecar.
        # A received but invalid verdict is a parsing outcome, not a transport failure.
        try:
            if response.status in ("incomplete", "failed", "cancelled"):
                raise ValueError(f"Judge generation status: {response.status}")
            judgment = {**parse_judgment(response.output_text or ""), "valid": True}
        except ValueError as exc:
            judgment = {"completed": None, "reason": f"Invalid judge verdict: {exc}"[:2000], "valid": False}
        return {
            **judgment,
            "model_server": reference.name,
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None,
            "method": "llm_judge",
            "response": response.model_dump(),
        }


class OwaspLlm06ResourcesServer(CompletionJudgeVerifier, SimpleResourcesServer):
    config: OwaspLlm06ResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)

    async def request_judge(self, params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        return await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )


if __name__ == "__main__":
    OwaspLlm06ResourcesServer.run_webserver()
