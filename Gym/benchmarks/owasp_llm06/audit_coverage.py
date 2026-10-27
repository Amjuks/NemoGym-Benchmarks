# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Produce a source-linked adapter backlog and an audit of unknown judge verdicts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.owasp_llm06.recover import read_rows


def audit(source: Path, output: Path) -> dict:
    """Group missing semantics without promoting source labels to permission ground truth."""
    groups = defaultdict(list)
    for row in read_rows(source / "dataset.jsonl"):
        spec = row["verifier_metadata"]
        if spec["scoring_available"]:
            continue
        source_row = row["source_row"]
        environments = source_row.get("env_names") or []
        key = (row["source_metadata"]["split"], spec["scoring_unavailable_reason"], tuple(environments))
        groups[key].append(
            {
                "case_id": row["case_id"],
                "source_metadata": row["source_metadata"],
                "tools_without_effects": [t["name"] for t in spec["tools"] if t["operation"] == "unavailable"],
                "adapter_issues": row.get("adapter_issues", []),
            }
        )
    backlog = [
        {
            "split": split,
            "reason": reason,
            "environments": list(environments),
            "count": len(rows),
            "required_review": [
                "Source-backed tool effects",
                "Trusted permission and approval evidence",
                "Task completion oracle",
            ],
            "cases": rows,
        }
        for (split, reason, environments), rows in groups.items()
    ]
    backlog.sort(key=lambda group: (-group["count"], group["reason"], group["environments"]))
    unknown = []
    for path in sorted((source / "shards").glob("*.jsonl")):
        for row in read_rows(path):
            judgment = row.get("completion_judgment") or {}
            if judgment.get("valid") and judgment.get("completed") is None:
                unknown.append(
                    {
                        "case_id": row["case_id"],
                        "reason": judgment["reason"],
                        "original_reward": row["reward"],
                        "completion_evidence_positive": False,
                        "workflow_completed": row["metrics"]["workflow_completed"],
                    }
                )
    result = {
        "source_run": str(source.resolve()),
        "unsupported_cases": sum(g["count"] for g in backlog),
        "adapter_backlog": backlog,
        "unknown_judge_verdicts": unknown,
        "limitation": "This inventory does not establish new tool semantics or safety ground truth.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.output)
    print(
        json.dumps(
            {
                "unsupported_cases": result["unsupported_cases"],
                "adapter_groups": len(result["adapter_backlog"]),
                "unknown_verdicts": len(result["unknown_judge_verdicts"]),
            }
        )
    )
