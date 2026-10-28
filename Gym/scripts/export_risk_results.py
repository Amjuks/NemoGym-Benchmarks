# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export whole-benchmark native scores grouped by risk; no task filtering or re-verification."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


if __package__:
    from .export_evaluation_results import final_text_present, identity, read_rows, write_csv
else:
    from export_evaluation_results import final_text_present, identity, read_rows, write_csv


def expected_cases(inputs: Path) -> set[tuple[int, int]]:
    """Include every task/repeat from the materialized full-benchmark inputs."""
    warnings = []
    selected = set()
    for row in read_rows(inputs, warnings):
        selected.add(identity(row))
    if not inputs.exists() or warnings or not selected:
        raise ValueError(f"{inputs}: full benchmark inputs unavailable or incomplete")
    return selected


def aggregate_saved(directory: Path, *, model: str, expected: set[tuple[int, int]], risk: str) -> dict:
    """Average all saved native rewards; final score requires every expected row."""
    warnings = []
    records = {}
    for record in read_rows(directory / "rollouts.jsonl", warnings):
        key = identity(record)
        if key not in expected:
            continue
        response = record.get("response") or {}
        records[key] = {
            "reward": record.get("reward"),
            "masked": record.get("mask_sample", False),
            "incomplete": response.get("status") == "incomplete",
            "empty": not final_text_present(response),
            "milestone": record.get("milestone_similarity"),
            "minefield": record.get("minefield_similarity"),
        }
    attempts = Counter()
    for failure in read_rows(directory / "rollouts_failures.jsonl", warnings):
        key = identity(failure)
        if key in expected:
            attempts[key] += 1
    scored = [
        record
        for record in records.values()
        if not record["masked"] and isinstance(record["reward"], (int, float)) and math.isfinite(record["reward"])
    ]

    def mean(field: str) -> float:
        values = [r[field] for r in scored if isinstance(r[field], (int, float)) and math.isfinite(r[field])]
        return sum(values) / len(values) if values else math.nan

    observed_score = mean("reward")
    row = {
        "model": model,
        "score": observed_score if len(scored) == len(expected) else math.nan,
        "observed_score": observed_score,
    }
    if risk == "llm06":
        row.update(
            {
                "mean_milestone_score": mean("milestone"),
                "mean_minefield_score": mean("minefield"),
                "minefield_hit_count": sum(r["minefield"] > 0 for r in scored if r["minefield"] is not None),
            }
        )
    row.update(
        {
            "success_count": sum(r["reward"] == 1 for r in scored),
            "partial_success_count": sum(0 < r["reward"] < 1 for r in scored),
            "failure_count": sum(r["reward"] == 0 for r in scored),
            "scored_count": len(scored),
            "completed_count": len(records),
            "expected_count": len(expected),
            "missing_count": len(expected) - len(records),
            "unscored_count": len(records) - len(scored),
            "completion_pct": 100 * len(records) / len(expected),
            "unrecovered_failure_count": len(attempts.keys() - records.keys()),
            "failed_request_attempts": sum(attempts.values()),
            "incomplete_response_count": sum(r["incomplete"] for r in records.values()),
            "empty_response_count": sum(r["empty"] for r in records.values()),
            "read_warning_count": len(warnings),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Evaluation directory containing manifest.json")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/risk_csv")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    output = args.output_dir or root / "risk_csv"
    output.mkdir(parents=True, exist_ok=True)
    rows = {"llm02": [], "llm06": []}
    selections = {}
    for entry in manifest["evaluations"]:
        risk = {"iheval": "llm02", "toolsandbox": "llm06"}.get(entry["benchmark"])
        if risk is None:
            continue
        directory = Path(entry["output"])
        if not directory.is_absolute():
            directory = root / directory
        selected = expected_cases(directory / "rollouts_materialized_inputs.jsonl")
        if len(selected) != entry["expected_rows"]:
            raise ValueError(f"{directory}: materialized row count does not match the full benchmark manifest")
        if risk in selections and selected != selections[risk]:
            raise ValueError(f"{risk}: models were not evaluated on the same task identities")
        selections[risk] = selected
        rows[risk].append(aggregate_saved(directory, model=entry["model"], expected=selected, risk=risk))
    for risk, values in rows.items():
        if values:
            write_csv(output / f"{risk}.csv", values, list(values[0]))
            print(
                f"{risk}: {len(values)} models, {len(selections[risk])} expected cases per model -> {output / (risk + '.csv')}"
            )
    generated = datetime.now(timezone.utc).isoformat()
    (output / "README.md").write_text(f"""# Results by risk

Generated (UTC): {generated}. Each CSV has one row per model; all metrics are numeric.

- **llm02.csv:** All **18,998 IHEval rows** per model, including every task and setting.
- **llm06.csv:** All **1,032 ToolSandbox scenarios** per model, including scenarios without minefields.

No risk-specific subset filtering is applied. Risk names group the benchmarks;
these are overall benchmark scores, not complete measurements of each OWASP risk.

## Calculations

| Column | Calculation / meaning |
|---|---|
| model | Evaluated model name. |
| score | Final full-benchmark average: sum of native rewards / expected_count. Only available when every expected row has a finite, unmasked score; otherwise NaN. Range 0–1; higher is better. |
| observed_score | Current sum of saved native rewards / scored_count. Provisional if any rows remain missing or unscored. Each scored row has equal weight. |
| success_count | Scored cases with reward = 1. |
| partial_success_count | Scored cases with 0 < reward < 1. |
| failure_count | Scored cases with reward = 0; not HTTP failures. |
| scored_count | Collected cases with a finite native reward, excluding masked cases. |
| completed_count | All benchmark cases with a saved result, including incomplete responses. |
| expected_count | Full benchmark size: 18,998 for IHEval or 1,032 for ToolSandbox. |
| missing_count | expected_count − completed_count. |
| unscored_count | completed_count − scored_count (masked or missing/nonfinite rewards). |
| completion_pct | 100 × completed_count / expected_count. Below 100 means provisional results. |
| unrecovered_failure_count | Benchmark cases with failed attempts and no saved result yet. |
| failed_request_attempts | All recorded failed attempts across the full benchmark, including recovered attempts. |
| incomplete_response_count | Collected responses marked incomplete by the API. |
| empty_response_count | Collected responses with no final output text. |
| read_warning_count | Unfinished final JSONL writes encountered; regenerate if nonzero. |
| mean_milestone_score | LLM06: sum of saved milestone scores / cases reporting that score. Higher is better. |
| mean_minefield_score | LLM06: sum of saved minefield scores / cases reporting that score. Lower is better. |
| minefield_hit_count | LLM06: scored cases with saved minefield score > 0. Lower is better. |

Missing cases never become zero scores. `NaN` means no scored data. Duplicate task/repeat
identities count once, using the last saved result. Native rewards are unchanged; empty or
incomplete responses retain their original scores and are counted separately for visibility.

General content-safety judging remains separate; it is not averaged into native benchmark rewards.
One full benchmark is grouped under each risk, so different benchmark scales are not averaged.
No LLM10 CSV is produced: this revision has no native resource-budget evaluation result.

Refresh from `Gym` using the existing Gym virtual environment:

```bash
.venv/bin/python scripts/export_risk_results.py results/native_owasp/{root.name}
```
""")


if __name__ == "__main__":
    main()
