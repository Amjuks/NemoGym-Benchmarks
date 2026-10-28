# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_evaluation_run import prepare_inputs


class PrepareInputsTests(unittest.TestCase):
    def test_chat_tool_conversion_preserves_grading_fields(self):
        row = {
            "task": "get-webpage",
            "setting": "conflict/default",
            "answer": {"task": "lang_detect", "content": "English"},
            "responses_create_params": {
                "input": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "fetch", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "page text"},
                ],
                "tools": [{"type": "function", "function": {"name": "fetch", "parameters": {"type": "object"}}}],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source.jsonl", Path(temporary) / "output.jsonl"
            source.write_text(json.dumps(row) + "\n")
            with (
                patch("benchmarks.iheval.prepare._SRC_TEST", source),
                patch("benchmarks.iheval.prepare._ensure_source"),
            ):
                self.assertEqual(prepare_inputs("iheval", output), 1)
            result = json.loads(output.read_text())
        self.assertEqual(result["answer"], row["answer"])
        self.assertEqual(result["task"], row["task"])
        self.assertEqual(result["setting"], row["setting"])
        self.assertEqual(result["responses_create_params"]["input"][0]["type"], "function_call")
        self.assertEqual(result["responses_create_params"]["input"][1]["type"], "function_call_output")
        self.assertEqual(result["responses_create_params"]["input"][1]["output"], "page text")
        self.assertEqual(result["responses_create_params"]["tools"][0]["name"], "fetch")

    def test_already_converted_responses_are_not_converted_again(self):
        params = {
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "fetch", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "page text"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source.jsonl", Path(temporary) / "output.jsonl"
            source.write_text(json.dumps({"responses_create_params": params, "answer": "gold"}) + "\n")
            with (
                patch("benchmarks.iheval.prepare._SRC_TEST", source),
                patch("benchmarks.iheval.prepare._ensure_source"),
            ):
                prepare_inputs("iheval", output)
            result = json.loads(output.read_text())
        self.assertEqual(result["responses_create_params"], params)
        self.assertEqual(result["answer"], "gold")
        self.assertEqual(result["_ng_task_index"], 0)
        self.assertEqual(result["agent_ref"]["name"], "iheval_benchmark_simple_agent")


if __name__ == "__main__":
    unittest.main()
