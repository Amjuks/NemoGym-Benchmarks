# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.export_risk_results import aggregate_saved, expected_cases


def save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class RiskResultsTests(unittest.TestCase):
    def test_all_iheval_tasks_settings_and_labels_are_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.jsonl"
            rows = [
                {"task": "translation", "setting": "conflict/test", "answer": "ordinary translation"},
                {"task": "system-prompt-extract", "setting": "reference/default", "answer": {"label": 0}},
                {"task": "system-prompt-extract", "setting": "conflict/strong_defense", "answer": {"label": 1}},
                {"task": "system-prompt-extract", "setting": "conflict/strong_defense", "answer": '{"label": 0}'},
                {"task": "system-prompt-extract", "setting": "conflict/weak_defense", "answer": {"label": 0}},
            ]
            save(path, [{"_ng_task_index": i, **row} for i, row in enumerate(rows)])
            self.assertEqual(expected_cases(path), {(i, 0) for i in range(5)})

    def test_toolsandbox_includes_scenarios_without_minefields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.jsonl"
            save(
                path,
                [
                    {"_ng_task_index": 0, "scenario": "generic_permission_named_task"},
                    {"_ng_task_index": 1, "scenario": "native_minefield_case"},
                ],
            )
            self.assertEqual(expected_cases(path), {(0, 0), (1, 0)})

    def test_pooling_uses_native_rewards_and_excludes_unrelated_failures_and_missing_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save(
                directory / "rollouts.jsonl",
                [
                    {"_ng_task_index": 0, "reward": 0, "minefield_similarity": 1, "milestone_similarity": 1},
                    {"_ng_task_index": 0, "reward": 1, "minefield_similarity": 0, "milestone_similarity": 1},
                    {"_ng_task_index": 1, "reward": 0.25, "minefield_similarity": 0, "milestone_similarity": 0.25},
                    {"_ng_task_index": 2, "reward": 0, "mask_sample": True},
                    {"_ng_task_index": 99, "reward": 0},
                ],
            )
            save(
                directory / "rollouts_failures.jsonl",
                [
                    {"_ng_task_index": 0},
                    {"_ng_task_index": 3},
                    {"_ng_task_index": 99},
                ],
            )
            row = aggregate_saved(directory, model="test", expected={(i, 0) for i in range(5)}, risk="llm06")
            self.assertEqual(row["observed_score"], 0.625)
            self.assertTrue(math.isnan(row["score"]))
            self.assertEqual(row["missing_count"], 2)
            self.assertEqual(row["unscored_count"], 1)
            self.assertEqual(row["scored_count"], 2)
            self.assertEqual(row["completed_count"], 3)
            self.assertEqual(row["completion_pct"], 60)
            self.assertEqual(row["success_count"], 1)
            self.assertEqual(row["partial_success_count"], 1)
            self.assertEqual(row["failure_count"], 0)
            self.assertEqual(row["failed_request_attempts"], 2)
            self.assertEqual(row["unrecovered_failure_count"], 1)
            self.assertEqual(row["minefield_hit_count"], 0)
            self.assertEqual(row["mean_milestone_score"], 0.625)
            self.assertTrue(
                math.isnan(aggregate_saved(directory, model="test", expected={(7, 0)}, risk="llm02")["score"])
            )
            complete = aggregate_saved(directory, model="test", expected={(0, 0), (1, 0)}, risk="llm06")
            self.assertEqual(complete["score"], 0.625)
            self.assertEqual(complete["score"], complete["observed_score"])


if __name__ == "__main__":
    unittest.main()
