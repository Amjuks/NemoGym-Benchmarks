# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a manifest's saved evaluation results without rerunning any verifier.

Writes scores.csv (one model/benchmark/scorer per row), metrics.csv (all saved
run-level aggregate and quality metrics), and rollouts.csv (task scores and
failure accounting). Aegis is a separate scorer, never mixed into native rewards.
Only the selected manifest is read; superseded runs are not combined.
Use --summary-only for one numeric model-by-metric table per benchmark/scorer.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def read_rows(path: Path, warnings: list[str]) -> Iterator[dict]:
    """Read a size-bounded JSONL snapshot, allowing an unfinished final write."""
    if not path.exists():
        return
    with path.open("rb") as stream:
        end = os.fstat(stream.fileno()).st_size
        number = 0
        while stream.tell() < end:
            line = stream.readline(end - stream.tell())
            number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if stream.tell() == end and not line.endswith(b"\n"):
                    warnings.append(f"{path.name}: unfinished final line {number}")
                    break
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def identity(row: dict) -> tuple[int, int]:
    """Use Gym's stable task/repeat identity, not the task's nonunique id field."""
    return int(row["_ng_task_index"]), int(row.get("_ng_rollout_index", 0))


def flatten(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Preserve saved scalar metrics, including lists, with explicit paths."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}/{index}")
    else:
        yield prefix, value


def load_optional(path: Path, warnings: list[str]) -> dict | list | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, (dict, list)):
            raise ValueError(f"{path}: expected a JSON object or array")
        return value
    except json.JSONDecodeError:
        warnings.append(f"{path.name}: JSON unavailable during write; rerun export")
        return None


def job_states(namespace: str | None) -> dict:
    if not namespace:
        return {}
    result = subprocess.run(
        ["kubectl", "get", "jobs", "-n", namespace, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    states = {}
    for job in json.loads(result.stdout)["items"]:
        status = job.get("status", {})
        conditions = {c["type"] for c in status.get("conditions", []) if c["status"] == "True"}
        state = "complete" if "Complete" in conditions else "failed" if "Failed" in conditions else "running"
        states[job["metadata"]["name"]] = {
            "job_status": state,
            "job_failed_pods": status.get("failed", 0),
            "job_started_at": status.get("startTime"),
            "job_completed_at": status.get("completionTime"),
        }
    return states


def final_text_present(response: dict) -> bool:
    return any(
        part.get("type") == "output_text" and str(part.get("text") or "").strip()
        for item in response.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def summarize(
    directory: Path, *, expected: set, common: dict, detail_stream: TextIO | None, detail_fields: set, metrics: list
) -> dict:
    warnings = []
    results = {}
    duplicate_rows = 0
    for record in read_rows(directory / "rollouts.jsonl", warnings):
        key = identity(record)
        duplicate_rows += key in results
        # Retain scores and diagnostics, never prompts, model text, or credentials.
        detail = {
            k: record.get(k) for k in ("task", "domain", "setting", "scenario", "failure_kind", "failure_reason")
        }
        for name, value in record.items():
            if not name.startswith("_") and name not in {"id", "task_idx"} and isinstance(value, (int, float)):
                detail[f"score/{name}"] = value
        for name in ("user_safety", "response_safety", "safety_categories"):
            if name in record:
                detail[name] = record[name]
        response = record.get("response") or {}
        detail["response_status"] = response.get("status")
        detail["response_incomplete_reason"] = (response.get("incomplete_details") or {}).get("reason")
        detail["response_error"] = bool(response.get("error"))
        detail["empty_final_text"] = not final_text_present(response)
        detail["response_model"] = response.get("model")
        for name, value in (response.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                detail[f"tokens/{name}"] = value
        judge_response = (record.get("judge_evaluation") or {}).get("response") or {}
        detail["judge_response_status"] = judge_response.get("status")
        detail["judge_incomplete_reason"] = (judge_response.get("incomplete_details") or {}).get("reason")
        for name, value in (judge_response.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                detail[f"judge_tokens/{name}"] = value
        results[key] = detail

    attempts = Counter()
    last_failures = {}
    failure_classes = Counter()
    for failure in read_rows(directory / "rollouts_failures.jsonl", warnings):
        key = identity(failure)
        attempts[key] += 1
        failure_classes[failure.get("_ng_failure_class", "unknown")] += 1
        last_failures[key] = {
            "last_failure_class": failure.get("_ng_failure_class"),
            "last_failure_type": failure.get("_ng_failure_type"),
            "last_failure_http_status": failure.get("_ng_failure_http_status"),
            "failure_marked_terminal": bool(failure.get("_ng_failure_terminal")),
        }

    valid = expected & results.keys()
    measured = [results[key] for key in valid if not results[key].get("score/mask_sample")]
    rewards = [d["score/reward"] for d in measured if isinstance(d.get("score/reward"), (int, float))]
    rewards = [v for v in rewards if math.isfinite(v)]
    counts = Counter()
    sums = Counter()
    denominators = Counter()
    for detail in measured:
        for name, value in detail.items():
            if name.startswith(("score/", "tokens/", "judge_tokens/")) and isinstance(value, (int, float)):
                if math.isfinite(value):
                    sums[name] += value
                    denominators[name] += 1
        counts["incomplete_responses"] += detail["response_status"] == "incomplete"
        counts["output_limit_responses"] += detail["response_incomplete_reason"] == "max_output_tokens"
        counts["response_errors"] += detail["response_error"]
        counts["empty_final_text_rows"] += detail["empty_final_text"]
        counts["unresolved_judge_rows"] += detail.get("score/resolved") is False
        counts["safe_judge_rows"] += detail.get("response_safety") == "safe"
        counts["unsafe_judge_rows"] += detail.get("response_safety") == "unsafe"
        counts["resolved_judge_rows"] += detail.get("score/resolved") is True
        counts["resolved_safe_judge_rows"] += (
            detail.get("score/resolved") is True and detail.get("response_safety") == "safe"
        )
        counts["resolved_unsafe_judge_rows"] += (
            detail.get("score/resolved") is True and detail.get("response_safety") == "unsafe"
        )
        counts["incomplete_judge_responses"] += detail["judge_response_status"] == "incomplete"

    for key in sorted(expected | results.keys() | attempts.keys()):
        if detail_stream is None:
            break
        present = key in results
        detail = {
            **common,
            "task_index": key[0],
            "rollout_index": key[1],
            "expected": key in expected,
            "record_status": "collected" if present else "failed" if key in attempts else "pending",
            "failure_entries": attempts[key],
            "recovered_failure": present and key in attempts,
            **last_failures.get(key, {}),
            **results.get(key, {}),
        }
        detail_fields.update(detail)
        detail_stream.write(json.dumps(detail) + "\n")

    recovered = attempts.keys() & valid
    unrecovered = (attempts.keys() & expected) - valid
    summary = {
        **common,
        "coverage_status": "complete" if valid == expected else "partial" if valid else "not_started",
        "expected_rows": len(expected),
        "collected_rows": len(valid),
        "missing_rows": len(expected - valid),
        "completion_pct": 100 * len(valid) / len(expected) if expected else None,
        "scored_rows": len(rewards),
        "masked_rows": len(valid) - len(measured),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "full_reward_rows": sum(v == 1 for v in rewards),
        "zero_reward_rows": sum(v == 0 for v in rewards),
        "partial_reward_rows": sum(0 < v < 1 for v in rewards),
        "failure_entries": sum(attempts.values()),
        "recovered_failure_rows": len(recovered),
        "unrecovered_failure_rows": len(unrecovered),
        "duplicate_rows": duplicate_rows,
        "unexpected_rows": len(results.keys() - expected),
        **{
            name: counts[name]
            for name in (
                "incomplete_responses",
                "output_limit_responses",
                "response_errors",
                "empty_final_text_rows",
                "unresolved_judge_rows",
                "safe_judge_rows",
                "unsafe_judge_rows",
                "incomplete_judge_responses",
                "resolved_judge_rows",
                "resolved_safe_judge_rows",
                "resolved_unsafe_judge_rows",
            )
        },
        **{f"observed_mean/{name}": sums[name] / denominators[name] for name in sorted(sums)},
        "failure_classes": dict(failure_classes),
        "rollouts_path": os.path.relpath(directory / "rollouts.jsonl"),
    }
    aggregate = load_optional(directory / "rollouts_aggregate_metrics.json", warnings)
    summary["native_aggregate_available"] = aggregate is not None
    aggregate_count = 0
    if aggregate is not None:
        for agent in aggregate:
            aggregate_count += sum(group.get("num_rollouts", 0) for group in agent.get("group_level_metrics", []))
            for scope in ("agent_metrics", "key_metrics", "repeat_level_metrics"):
                for name, value in flatten(agent.get(scope, {})):
                    metrics.append(
                        {
                            **common,
                            "source": "native_aggregate",
                            "agent": agent.get("agent_ref", {}).get("name"),
                            "metric": f"{scope}/{name}",
                            "value": value,
                            "source_path": os.path.relpath(directory / "rollouts_aggregate_metrics.json"),
                        }
                    )
        summary["native_aggregate_rows"] = aggregate_count
        summary["native_aggregate_matches_collected"] = aggregate_count == len(valid)
    quality = load_optional(directory / "quality_summary.json", warnings)
    if quality is not None:
        for name, value in flatten(quality.get("run", {})):
            metrics.append(
                {
                    **common,
                    "source": "native_quality",
                    "metric": name,
                    "value": value,
                    "source_path": os.path.relpath(directory / "quality_summary.json"),
                }
            )
        summary["quality_snapshot_rows"] = quality.get("run", {}).get("artifacts", {}).get("records")
        summary["quality_snapshot_matches_collected"] = summary["quality_snapshot_rows"] == len(valid)
        summary["quality_snapshot_healthy"] = quality.get("run", {}).get("artifacts", {}).get("healthy")
        for verdict, count in quality.get("run", {}).get("verdicts", {}).items():
            summary[f"quality_verdict/{verdict}"] = count
    summary["read_warnings"] = warnings
    return summary


def write_csv(path: Path, rows, fields: list[str]) -> None:
    """Replace an earlier export only after its replacement is fully written."""
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def write_benchmark_summaries(output: Path, summaries: list[dict]) -> None:
    """Write model-by-metric tables; missing measurements remain numeric NaN."""
    common = {
        "full_success_count": "full_reward_rows",
        "partial_success_count": "partial_reward_rows",
        "zero_score_count": "zero_reward_rows",
        "completed_count": "collected_rows",
        "expected_count": "expected_rows",
        "completion_pct": "completion_pct",
        "unrecovered_failure_count": "unrecovered_failure_rows",
        "failed_request_attempts": "failure_entries",
        "incomplete_response_count": "incomplete_responses",
        "context_limit_count": "output_limit_responses",
        "mean_input_tokens": "observed_mean/tokens/input_tokens",
        "mean_output_tokens": "observed_mean/tokens/output_tokens",
    }
    specific = {
        "iheval": {
            "mean_strict_score": "observed_mean/score/strict",
            "mean_loose_score": "observed_mean/score/loose",
        },
        "toolsandbox": {
            "mean_milestone_score": "observed_mean/score/milestone_similarity",
            "mean_minefield_score": "observed_mean/score/minefield_similarity",
            "mean_turn_count": "observed_mean/score/turn_count",
        },
    }
    for scorer in ("iheval", "toolsandbox", "aegis_v4_safety"):
        rows = []
        for summary in summaries:
            if summary["scorer"] != scorer:
                continue
            mapping = {"mean_score": "mean_reward", **specific.get(scorer, {}), **common}
            if scorer == "aegis_v4_safety":
                mapping = {
                    "mean_score": "mean_reward",
                    "resolved_count": "resolved_judge_rows",
                    "unresolved_count": "unresolved_judge_rows",
                    "safe_count": "resolved_safe_judge_rows",
                    "unsafe_count": "resolved_unsafe_judge_rows",
                    **{
                        key: common[key]
                        for key in (
                            "completed_count",
                            "expected_count",
                            "completion_pct",
                            "unrecovered_failure_count",
                            "failed_request_attempts",
                        )
                    },
                }
            row = {"model": summary["model"]}
            if scorer == "aegis_v4_safety":
                resolved = summary["resolved_judge_rows"]
                row.update(
                    {
                        "mean_score": summary.get("mean_reward"),
                        "safe_pct_resolved": 100 * summary["resolved_safe_judge_rows"] / resolved
                        if resolved
                        else math.nan,
                        "unsafe_pct_resolved": 100 * summary["resolved_unsafe_judge_rows"] / resolved
                        if resolved
                        else math.nan,
                    }
                )
            row.update({name: summary.get(source) for name, source in mapping.items()})
            row = {key: math.nan if value is None else value for key, value in row.items()}
            rows.append(row)
        if rows:
            filename = "safety_judge_summary.csv" if scorer == "aegis_v4_safety" else f"{scorer}_summary.csv"
            write_csv(output / filename, rows, list(rows[0]))
            print(f"Wrote {len(rows)} model rows to {output / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing manifest.json")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/csv")
    parser.add_argument("--namespace", help="Optionally read current Job status using kubectl")
    parser.add_argument("--summary-only", action="store_true", help="Write only the three numeric model-summary CSVs")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    output = args.output_dir or root / "csv"
    output.mkdir(parents=True, exist_ok=True)
    states = job_states(args.namespace)
    generated = datetime.now(timezone.utc).isoformat()
    summaries, metrics, detail_fields = [], [], set()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as details:
        for entry in manifest["evaluations"]:
            directory = Path(entry["output"])
            if not directory.is_absolute():
                directory = root / directory
            warnings = []
            inputs = directory / "rollouts_materialized_inputs.jsonl"
            expected = {identity(row) for row in read_rows(inputs, warnings)}
            if not expected:
                # These manifests declare one repeat and consecutive task indices.
                expected = {(index, 0) for index in range(entry["expected_rows"])}
            if len(expected) != entry["expected_rows"] or warnings:
                raise ValueError(f"{inputs}: incomplete inputs or manifest count mismatch")
            for scorer, location in [(entry["benchmark"], directory)] + (
                [("aegis_v4_safety", directory / "aegis")] if entry["benchmark"] == "iheval" else []
            ):
                common = {
                    "generated_at_utc": generated,
                    "run_id": root.name,
                    "model": entry["model"],
                    "benchmark": entry["benchmark"],
                    "scorer": scorer,
                    "judge_model": manifest.get("judge") if scorer == "aegis_v4_safety" else None,
                    "job": entry.get("job"),
                    **states.get(entry.get("job"), {"job_status": "unknown"}),
                }
                summaries.append(
                    summarize(
                        location,
                        expected=expected,
                        common=common,
                        detail_stream=None if args.summary_only else details,
                        detail_fields=detail_fields,
                        metrics=metrics,
                    )
                )
        write_benchmark_summaries(output, summaries)
        if args.summary_only:
            return
        write_csv(output / "scores.csv", summaries, list(dict.fromkeys(k for row in summaries for k in row)))
        metric_fields = list(dict.fromkeys(k for row in metrics for k in row)) or [
            "model",
            "benchmark",
            "scorer",
            "source",
            "metric",
            "value",
            "source_path",
        ]
        write_csv(output / "metrics.csv", metrics, metric_fields)
        details.seek(0)
        write_csv(output / "rollouts.csv", (json.loads(line) for line in details), sorted(detail_fields))
    for row in summaries:
        if row["scorer"] != "aegis_v4_safety":
            print(
                f"{row['model']:24} {row['benchmark']:12} {row['collected_rows']:5}/{row['expected_rows']} "
                f"{row['job_status']:8} unrecovered_failures={row['unrecovered_failure_rows']}"
            )
    print(f"Wrote {len(summaries)} summary rows to {output / 'scores.csv'}")
    print(f"Wrote {len(metrics)} saved run-level metrics to {output / 'metrics.csv'}")
    print(f"Task scores, missing rows, and failure history: {output / 'rollouts.csv'}")


if __name__ == "__main__":
    main()
