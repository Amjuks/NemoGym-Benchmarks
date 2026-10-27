# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent standalone judge using Gym's external OpenAI-compatible client."""

import asyncio
from pathlib import Path

from aiohttp import ClientResponseError
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, SecretStr

from benchmarks.response_utils import normalize_usage
from nemo_gym.judge import JudgeError, reraise_judge_errors
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.owasp_llm06.app import CompletionJudgeConfig, CompletionJudgeVerifier


class JudgeEndpoint(BaseModel):
    openai_base_url: str
    openai_api_key: SecretStr
    openai_model: str
    upstream_request_timeout_seconds: float = Field(default=120, gt=0)
    max_http_attempts: int = Field(default=3, ge=1)
    max_rate_limit_wait_seconds: float = Field(default=120, ge=0)


class StandaloneVerifier(CompletionJudgeVerifier):
    def __init__(self, config: CompletionJudgeConfig, endpoint: JudgeEndpoint):
        self.config = config
        self.endpoint = endpoint
        self._judge_semaphore = asyncio.Semaphore(config.judge_endpoint_max_concurrency)
        self.client = NeMoGymAsyncOpenAI(
            base_url=endpoint.openai_base_url.rstrip("/"),
            api_key=endpoint.openai_api_key.get_secret_value(),
            request_timeout_seconds=endpoint.upstream_request_timeout_seconds,
            max_http_attempts=endpoint.max_http_attempts,
            max_rate_limit_wait_seconds=endpoint.max_rate_limit_wait_seconds,
            max_connection_retries=0,
        )

    @classmethod
    def from_config(cls, path: Path, *, env_file: str | Path | None = None) -> "StandaloneVerifier":
        """Read Gym-style judge settings without starting or discovering any Gym servers."""
        config = OmegaConf.load(path)
        if env_file is not None:
            env_path = Path(env_file)
            if env_path.is_file():
                config = OmegaConf.merge(config, OmegaConf.load(env_path))
            elif str(env_path) != "env.yaml":
                raise FileNotFoundError(f"Judge environment file does not exist: {env_path}")
        resources = config.owasp_llm06_resources_server.resources_servers.owasp_llm06
        judge_config = CompletionJudgeConfig.model_validate(OmegaConf.to_container(resources, resolve=True))
        if judge_config.judge_model_server is None:
            raise ValueError("Judge config needs an independent judge_model_server reference")
        model = config[judge_config.judge_model_server.name].responses_api_models.openai_model
        endpoint = JudgeEndpoint.model_validate(OmegaConf.to_container(model, resolve=True))
        return cls(judge_config, endpoint)

    def profile(self) -> dict:
        """Credential-free grading settings used by run metadata and resume checks."""
        return {
            "transport": "standalone_nemo_gym_client",
            "model_server": self.config.judge_model_server.name,
            "model": self.endpoint.openai_model,
            "base_url": self.endpoint.openai_base_url,
            "parameters": self.config.judge_responses_create_params.model_dump(exclude_unset=True),
            "system_message": self.config.judge_system_message,
        }

    async def request_judge(self, params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        async def request() -> NeMoGymResponse:
            payload = params.model_dump(exclude_unset=True)
            payload.update(model=self.endpoint.openai_model, store=False, stream=False)
            for attempt in range(self.endpoint.max_http_attempts):
                try:
                    raw = await self.client.create_response(**payload)
                    break
                except ClientResponseError as exc:
                    content = getattr(exc, "response_content", b"")
                    detail = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                    # The gateway intermittently rejects its upstream header, even for identical requests.
                    # Other 400 responses are permanent request errors and must not be retried here.
                    transient_header = exc.status == 400 and "could not decode header" in detail.lower()
                    if transient_header and attempt + 1 < self.endpoint.max_http_attempts:
                        await asyncio.sleep(min(2**attempt, 4))
                        continue
                    # Never persist request headers or credentials echoed by a provider.
                    detail = (detail or exc.message).replace(
                        self.endpoint.openai_api_key.get_secret_value(), "[REDACTED]"
                    )
                    raise JudgeError(f"HTTP {exc.status}: {detail[:8000]}") from exc
            # This gateway returns Chat-style token names even on /responses.
            # Normalize the envelope before validating Gym's Responses schema.
            return NeMoGymResponse.model_validate(normalize_usage(raw))

        return await reraise_judge_errors(request())
