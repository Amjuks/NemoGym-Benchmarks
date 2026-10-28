# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reporting checks: coverage and failed requests must not become passing scores."""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.export_evaluation_results import read_rows, summarize


def result(index: int, reward: float, **extra) -> dict:
    return {
        "_ng_task_index": index,
        "_ng_rollout_index": 0,
        "reward": reward,
        "response": {"status": "completed", "output": []},
        **extra,
    }


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ExportResultsTests(unittest.TestCase):
    def test_partial_coverage_recovered_failures_and_masked_scores(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryFile(mode="w+") as details:
            directory = Path(temporary)
            save_rows(
                directory / "rollouts.jsonl",
                [
                    result(0, 0.25),
                    result(0, 1.0),  # Repeated identity is flagged and counted once, using the latest record.
                    result(1, 0.0, mask_sample=True),
                    result(
                        2,
                        0.5,
                        response={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
                    ),
                ],
            )
            save_rows(
                directory / "rollouts_failures.jsonl",
                [
                    {"_ng_task_index": 0, "_ng_failure_class": "agent_run_error"},
                    {"_ng_task_index": 0, "_ng_failure_class": "agent_run_error"},
                    {"_ng_task_index": 3, "_ng_failure_class": "timeout_exceeded"},
                ],
            )
            aggregate = [{"agent_metrics": {"mean/reward": 1.0}, "group_level_metrics": [{"num_rollouts": 1}]}]
            (directory / "rollouts_aggregate_metrics.json").write_text(json.dumps(aggregate))
            metrics = []
            report = summarize(
                directory,
                expected={(i, 0) for i in range(5)},
                common={},
                detail_stream=details,
                detail_fields=set(),
                metrics=metrics,
            )
            self.assertEqual((report["collected_rows"], report["missing_rows"]), (3, 2))
            self.assertEqual(report["coverage_status"], "partial")
            self.assertEqual(report["mean_reward"], 0.75)
            self.assertEqual(report["full_reward_rows"], 1)
            self.assertEqual(report["partial_reward_rows"], 1)
            self.assertEqual(report["zero_reward_rows"], 0)
            self.assertEqual(report["masked_rows"], 1)
            self.assertEqual(report["duplicate_rows"], 1)
            self.assertEqual(report["failure_entries"], 3)
            self.assertEqual(report["recovered_failure_rows"], 1)
            self.assertEqual(report["unrecovered_failure_rows"], 1)
            self.assertEqual(report["output_limit_responses"], 1)
            self.assertFalse(report["native_aggregate_matches_collected"])
            self.assertEqual(metrics[0]["value"], 1.0)  # Saved aggregate is preserved with its stale-count flag.
            details.seek(0)
            rows = [json.loads(line) for line in details]
            self.assertEqual([row["record_status"] for row in rows], ["collected"] * 3 + ["failed", "pending"])

    def test_aegis_unresolved_is_visible_and_not_mislabeled_as_unsafe(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryFile(mode="w+") as details:
            directory = Path(temporary)
            save_rows(
                directory / "rollouts.jsonl",
                [
                    result(0, 1, resolved=True, response_safety="safe"),
                    result(1, 0, resolved=True, response_safety="unsafe"),
                    result(2, 0, resolved=False, response_safety="unsafe", failure_reason="judge omitted user label"),
                ],
            )
            report = summarize(
                directory,
                expected={(i, 0) for i in range(3)},
                common={},
                detail_stream=details,
                detail_fields=set(),
                metrics=[],
            )
            self.assertEqual(report["coverage_status"], "complete")
            self.assertEqual(report["unresolved_judge_rows"], 1)
            self.assertEqual(report["unsafe_judge_rows"], 2)
            self.assertEqual(report["safe_judge_rows"], 1)
            self.assertEqual(report["resolved_judge_rows"], 2)
            self.assertEqual(report["resolved_unsafe_judge_rows"], 1)

    def test_incomplete_final_write_is_warned_but_bad_middle_line_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollouts.jsonl"
            path.write_text('{"_ng_task_index": 0}\n{"_ng_task_index":')
            warnings = []
            self.assertEqual(len(list(read_rows(path, warnings))), 1)
            self.assertEqual(len(warnings), 1)
            path.write_text('{"_ng_task_index": 0}\nBROKEN\n{}\n')
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_rows(path, []))

    def test_cli_exports_all_scorers_without_judging_or_reading_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iheval = root / "model/iheval"
            tools = root / "model/toolsandbox"
            save_rows(iheval / "rollouts_materialized_inputs.jsonl", [{"_ng_task_index": i} for i in range(2)])
            save_rows(iheval / "rollouts.jsonl", [result(0, 1, strict=1.0)])
            save_rows(tools / "rollouts.jsonl", [])
            manifest = {
                "judge": "nemotron-3-content-safety",
                "evaluations": [
                    {"model": "test-model", "benchmark": "iheval", "expected_rows": 2, "output": str(iheval)},
                    {"model": "test-model", "benchmark": "toolsandbox", "expected_rows": 2, "output": str(tools)},
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            script = Path(__file__).resolve().parents[1] / "export_evaluation_results.py"
            subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)
            with (root / "csv/scores.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["scorer"] for row in rows], ["iheval", "aegis_v4_safety", "toolsandbox"])
            self.assertEqual(rows[0]["completion_pct"], "50.0")
            self.assertEqual(rows[1]["judge_model"], "nemotron-3-content-safety")
            self.assertEqual(rows[1]["mean_reward"], "")
            self.assertEqual(rows[2]["coverage_status"], "not_started")
            with (root / "csv/rollouts.csv").open() as stream:
                details = list(csv.DictReader(stream))
            self.assertEqual(len(details), 6)
            self.assertEqual(details[0]["score/strict"], "1.0")
            subprocess.run([sys.executable, str(script), str(root), "--summary-only"], check=True, capture_output=True)
            for filename in ("iheval_summary.csv", "toolsandbox_summary.csv", "safety_judge_summary.csv"):
                with (root / "csv" / filename).open() as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames[:2], ["model", "mean_score"])
                    compact = list(reader)
                self.assertEqual(len(compact), 1)
                for key, value in compact[0].items():
                    if key != "model":
                        float(value)  # Every metric cell must be numeric, including NaN for unavailable scores.
                if filename == "iheval_summary.csv":
                    self.assertEqual(compact[0]["mean_score"], "1.0")
                    self.assertEqual(compact[0]["completion_pct"], "50.0")
                else:
                    self.assertTrue(math.isnan(float(compact[0]["mean_score"])))


if __name__ == "__main__":
    unittest.main()
