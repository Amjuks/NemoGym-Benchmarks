#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge exactly one completed output per indexed shard and compute a global report."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stats(items: list[dict]) -> dict:
    scored = [r for r in items if r.get("reward") is not None]
    return {
        "total": len(items),
        "scored": len(scored),
        "errors": len(items) - len(scored),
        "mean_reward": round(sum(r["reward"] for r in scored) / len(scored), 5) if scored else None,
        "zero_reward_cases": sum(r["reward"] == 0 for r in scored),
    }


def aggregate(
    folder: Path,
    *,
    shards: int,
    risk: str,
    input_path: Path | None = None,
    expected_count: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    if shards < 1:
        raise ValueError("Shard count must be positive")
    shard_folder = folder / "shards" if (folder / "shards").is_dir() else folder
    records = []
    seen = set()
    for index in range(shards):
        path = shard_folder / f"shard-{index}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing completed shard: {path}")
        # JSONL records end at LF. Unicode separators inside JSON strings are data.
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("case_id")
            if not key or key in seen:
                raise ValueError(f"Missing or duplicate case_id: {key}")
            if (
                risk in ("llm06", "llm10")
                and row.get("reward") is not None
                and not isinstance(row.get("metrics"), dict)
            ):
                raise ValueError(f"Missing {risk} metrics: {key}")
            if risk == "llm06" and row.get("scoring_available") is False and row.get("reward") is not None:
                raise ValueError(f"Reward present for case without a scoring adapter: {key}")
            seen.add(key)
            records.append(row)
    records.sort(key=lambda x: str(x["case_id"]))
    if input_path:
        input_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        expected = {row["case_id"] for row in input_rows}
        if len(expected) != len(input_rows):
            raise ValueError("Input contains duplicate case IDs")
        if seen != expected:
            raise ValueError(
                f"Input/result IDs differ: {len(expected - seen)} missing, {len(seen - expected)} unexpected"
            )
        if risk == "llm06":
            unavailable = {
                row["case_id"] for row in input_rows if row["verifier_metadata"].get("scoring_available") is False
            }
            for row in records:
                if row["case_id"] in unavailable and (
                    row.get("scoring_available") is not False or row.get("reward") is not None
                ):
                    raise ValueError(f"Result contradicts input scoring availability: {row['case_id']}")
    elif expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} unique cases, got {len(records)}")
    if risk == "llm10":
        from benchmarks.owasp_llm10.reporting import build_summary, markdown

        if any(
            len({json.dumps(r.get(k), sort_keys=True) for r in records}) > 1
            for k in ("model", "run_id", "run_parameters")
        ):
            raise ValueError("Cannot combine different LLM10 models, run IDs or execution settings")
        from resources_servers.owasp_llm10.app import evaluate
        from resources_servers.owasp_llm10.budget import CaseSpec, Telemetry

        input_by_id = {r["case_id"]: r for r in input_rows} if input_path else {}
        for row in records:
            if row.get("telemetry") is None:
                if row.get("reward") is not None:
                    raise ValueError(f"Scored LLM10 result missing telemetry: {row['case_id']}")
                continue
            spec = (input_by_id.get(row["case_id"]) or {}).get("verifier_metadata")
            if spec is None:
                # Budget and expectations are stored in the result for deterministic replay.
                spec = row["case_spec"]
            expected_metrics = evaluate(CaseSpec.model_validate(spec), Telemetry.model_validate(row["telemetry"]))
            if row.get("reward") != expected_metrics["reward"] or row.get("metrics") != expected_metrics:
                raise ValueError(f"LLM10 result disagrees with telemetry: {row['case_id']}")
        preparation_path = (input_path.parent if input_path else folder) / "preparation_report.json"
        preparation = json.loads(preparation_path.read_text()) if preparation_path.is_file() else None
        report = build_summary(records, preparation_report=preparation)
        report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path else None
        report_text = markdown(report, records)
    elif risk == "llm06":
        from benchmarks.owasp_llm06.reporting import build_summary, markdown

        if len({r.get("benchmark_version", "1") for r in records}) > 1 or len({r["model"] for r in records}) > 1:
            raise ValueError("Cannot combine different benchmark versions or models in one report")
        if len({json.dumps(r.get("run_parameters", {}).get("judge_profile"), sort_keys=True) for r in records}) > 1:
            raise ValueError("Cannot combine different judge profiles in one report")
        preparation = None
        preparation_path = input_path.parent / "preparation_report.json" if input_path else None
        if preparation_path and preparation_path.is_file():
            preparation = json.loads(preparation_path.read_text())
        report = build_summary(records, preparation_report=preparation)
        report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path else None
        report_text = markdown(report, records)
    else:
        groups = defaultdict(list)
        for row in records:
            groups[row.get("source_dataset") or "unknown"].append(row)
        report = {
            "overall": stats(records),
            "by_dataset": {k: stats(v) for k, v in sorted(groups.items())},
            "warning": "Verifier reward is not synonymous with leakage rate.",
        }
        report_text = None
    # Validate completeness before publishing any aggregate output.
    folder = output_dir or folder
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "all_results.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report_text is not None:
        (folder / "report.md").write_text(report_text, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("/results/llm02"))
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--risk", choices=("llm02", "llm06", "llm10"), default="llm02")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--output-dir", type=Path, help="Write reports to a writable folder, keeping source shards unchanged"
    )
    args = parser.parse_args()
    if args.risk in ("llm06", "llm10") and args.input is None and args.expected_count is None:
        snapshot = args.results_dir / "dataset.jsonl"
        args.input = (
            snapshot
            if snapshot.is_file()
            else Path(__file__).parent / f"benchmarks/owasp_{args.risk}/data/prepared.jsonl"
        )
    count = args.expected_count if args.expected_count is not None else (3232 if args.risk == "llm02" else None)
    print(
        json.dumps(
            aggregate(
                args.results_dir,
                shards=args.shards,
                risk=args.risk,
                input_path=args.input,
                expected_count=count,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )
