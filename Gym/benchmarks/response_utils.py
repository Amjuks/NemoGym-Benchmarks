# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize OpenAI-compatible response envelopes for the OWASP runners and judges."""


def normalize_usage(response: dict) -> dict:
    """Normalize OpenAI-compatible gateway usage for NeMo Gym's Responses schema."""
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    response["usage"] = {
        **usage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "input_tokens_details": {
            **input_details,
            "cached_tokens": input_details.get("cached_tokens", prompt_details.get("cached_tokens", 0)),
        },
        "output_tokens_details": {
            **output_details,
            "reasoning_tokens": output_details.get(
                "reasoning_tokens", completion_details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
            ),
        },
    }
    return response
