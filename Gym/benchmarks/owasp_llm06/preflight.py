# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject full benchmark launches when prepared cases lack scoring contracts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from resources_servers.owasp_llm06.mock_tools import CaseSpec


def inspect_dataset(path: Path, *, expected_count: int = 4220) -> dict:
    """Check necessary scoring prerequisites, without claiming oracle correctness."""
    seen = set()
    blocked = []
    invalid = []
    reasons = Counter()
    versions = set()
    total = 0
    no_source_tools = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                identity = row["case_id"]
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError("Missing or duplicate case ID")
                seen.add(identity)
                versions.add(row["benchmark_version"])
                spec = CaseSpec.model_validate(row["verifier_metadata"])
                if spec.benchmark_version != row["benchmark_version"]:
                    raise ValueError("Case/spec benchmark versions differ")
                source = row.get("source_row", {})
                if row.get("source_metadata", {}).get("split") == "train":
                    environments = json.loads(source["environments"])
                    if not any(env.get("tools") for env in environments):
                        no_source_tools.append(identity)
                if not spec.scoring_available:
                    reason = spec.scoring_unavailable_reason
                elif not spec.tools or not spec.policy.evidence:
                    reason = "missing_tool_or_policy_evidence"
                else:
                    reason = ""
                if reason:
                    reasons[reason] += 1
                    blocked.append(
                        {"case_id": identity, "reason": reason, "source_metadata": row.get("source_metadata")}
                    )
            except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as exc:
                invalid.append({"line": line_number, "error": str(exc)[:500]})
    ready = total == expected_count and len(seen) == total and len(versions) == 1 and not blocked and not invalid
    return {
        "ready": ready,
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_cases": expected_count,
        "total_cases": total,
        "supported_cases": total - len(blocked) - len(invalid),
        "blocked_cases": blocked,
        "blocked_reason_counts": dict(reasons.most_common()),
        "invalid_records": invalid,
        "benchmark_versions": sorted(versions),
        "source_cases_without_declared_tools": no_source_tools,
        "limitation": "Passing checks necessary metadata only; it does not independently validate permission or completion ground truth.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=4220)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.expected_cases < 1:
        parser.error("Expected case count must be positive")
    report = inspect_dataset(args.input, expected_count=args.expected_cases)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("ready", "total_cases", "expected_cases", "supported_cases", "blocked_reason_counts")
            },
            indent=2,
        )
    )
    if not report["ready"]:
        print("Full run blocked: complete and review scoring contracts for every case before launch.")
        raise SystemExit(2)
