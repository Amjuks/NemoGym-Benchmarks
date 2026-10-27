# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-run launch gates reject incomplete, malformed and inconsistent inputs."""

import json
from pathlib import Path

from benchmarks.owasp_llm06.preflight import inspect_dataset

from .test_app import ROOT


def test_readiness_requires_every_case_and_scoring_contract(tmp_path: Path):
    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().split("\n")[0])
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert inspect_dataset(path, expected_count=1)["ready"]
    assert not inspect_dataset(path, expected_count=2)["ready"]
    row["verifier_metadata"]["scoring_available"] = False
    row["verifier_metadata"]["scoring_unavailable_reason"] = "missing permission oracle"
    path.write_text(json.dumps(row) + "\n")
    report = inspect_dataset(path, expected_count=1)
    assert not report["ready"] and report["supported_cases"] == 0
    assert report["blocked_reason_counts"] == {"missing permission oracle": 1}


def test_readiness_rejects_duplicate_and_invalid_records(tmp_path: Path):
    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().split("\n")[0])
    path = tmp_path / "dataset.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2)
    report = inspect_dataset(path, expected_count=2)
    assert not report["ready"] and "duplicate" in report["invalid_records"][0]["error"]
    path.write_text('{"case_id": "broken"}\n')
    assert not inspect_dataset(path, expected_count=1)["ready"]
    path.write_text("")
    assert not inspect_dataset(path, expected_count=1)["ready"]


def test_readiness_preserves_unicode_and_requires_policy_evidence(tmp_path: Path):
    row = json.loads((ROOT / "benchmarks/owasp_llm06/data/smoke.jsonl").read_text().split("\n")[0])
    row["responses_create_params"]["input"] = "Task with \u2028 a Unicode separator"
    row["verifier_metadata"]["policy"]["evidence"] = []
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    report = inspect_dataset(path, expected_count=1)
    assert report["total_cases"] == 1 and not report["ready"]
    assert report["blocked_reason_counts"] == {"missing_tool_or_policy_evidence": 1}
