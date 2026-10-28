#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed OWASP runner sharing transport, checkpoints and sharding across risks."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

from benchmarks.response_utils import normalize_usage
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    is_global_aiohttp_client_setup,
    set_global_aiohttp_client,
)
from nemo_gym.server_utils import request as gym_request
from resources_servers.owasp_llm02.app import OwaspLlm02Verifier, OwaspLlm02VerifyRequest


if TYPE_CHECKING:
    from resources_servers.owasp_llm06.app import OwaspLlm06Verifier
    from resources_servers.owasp_llm06.standalone import StandaloneVerifier
    from resources_servers.owasp_llm10.app import OwaspLlm10Verifier


def extract_text(response: dict) -> str:
    return "\n".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def load_rows(path: str | Path, shard: int, shards: int, limit: int) -> list[dict]:
    selected = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            # Stable physical-row sharding across identical input copies.
            if (line_no - 1) % shards != shard:
                continue
            selected.append(row)
            if limit and len(selected) >= limit:
                break
    return selected


class Gateway:
    """Shared aiohttp transport with the existing retry and timeout settings."""

    def __init__(self, args: argparse.Namespace):
        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(
                GlobalAIOHTTPAsyncClientConfig(
                    global_aiohttp_connector_limit=max(32, args.concurrency * 2),
                    global_aiohttp_connector_limit_per_host=max(32, args.concurrency * 2),
                )
            )
        self.args = args
        self.attempts = 0

    async def send(self, payload: dict) -> dict:
        # LLM10 accounts every network dispatch itself; transport retries would hide consumption.
        retries = 0 if self.args.risk == "llm10" else self.args.retries
        for attempt in range(retries + 1):
            self.attempts += 1
            try:
                response = await gym_request(
                    "POST",
                    self.args.base_url.rstrip("/") + "/responses",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.args.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self.args.timeout),
                    _max_num_tries=1,
                    _max_connection_retries=0,
                )
                async with response:
                    detail = await response.text()
                    if response.status >= 400:
                        retryable = response.status in (408, 429, 500, 502, 503, 504) or (
                            response.status == 400 and "could not decode header" in detail.lower()
                        )
                        if not retryable:
                            raise RuntimeError(f"HTTP {response.status}: {detail[:1500]}")
                        raise aiohttp.ClientResponseError(
                            response.request_info, response.history, status=response.status, message=detail[:1500]
                        )
                    body = json.loads(detail)
                    if body.get("error"):
                        raise RuntimeError(str(body["error"]))
                    return body if self.args.risk == "llm10" else normalize_usage(body)
            except (TimeoutError, aiohttp.ClientError):
                if attempt == retries:
                    raise
                await asyncio.sleep(min(30, 2**attempt + random.random()))
        raise RuntimeError("No request attempts configured")


async def process(
    row: dict,
    verifier: "OwaspLlm02Verifier | OwaspLlm06Verifier | StandaloneVerifier | OwaspLlm10Verifier",
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        started = time.monotonic()
        record = {
            "case_id": row.get("case_id"),
            "source_dataset": row.get("source_dataset"),
            "scenario_type": row.get("scenario_type"),
            "authorized": row.get("authorized"),
            "source_metadata": row.get("source_metadata"),
            "model": args.model,
            "shard_index": args.shard_index,
            "reward": None,
            "model_response": None,
            "usage": None,
            "error": None,
            "run_parameters": {
                key: getattr(args, key)
                for key in (
                    "risk",
                    "base_url",
                    "max_output_tokens",
                    "timeout",
                    "retries",
                    "concurrency",
                    "shard_count",
                    "max_turns",
                    "max_tool_calls",
                )
            },
        }
        if args.risk == "llm06":
            record["benchmark_version"] = row.get("benchmark_version", "1")
            record["scoring_available"] = row["verifier_metadata"].get("scoring_available", True)
            record["scoring_unavailable_reason"] = row["verifier_metadata"].get("scoring_unavailable_reason", "")
            record["run_parameters"]["judge_profile"] = getattr(args, "judge_profile", None)
        gateway = Gateway(args)
        try:
            payload = {
                "model": args.model,
                "input": row["responses_create_params"]["input"],
                "max_output_tokens": args.max_output_tokens,
                "store": False,
                "stream": False,
            }
            if args.risk == "llm10":
                from resources_servers.owasp_llm10.standalone import execute

                record.update(await execute(row, gateway.send, payload, verifier=verifier))
                record["run_parameters"]["retries"] = 0
                record["run_parameters"].pop("max_turns")
                record["run_parameters"].pop("max_tool_calls")
                record["run_id"] = getattr(args, "run_id", "run-001")
                record["task_type"] = row.get("task_type")
                record["attempts"] = gateway.attempts
                record["duration_seconds"] = round(time.monotonic() - started, 3)
                return record
            elif args.risk == "llm06":
                from resources_servers.owasp_llm06.agent import rollout
                from resources_servers.owasp_llm06.app import OwaspLlm06VerifyRequest

                result = await rollout(
                    row, gateway.send, payload, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls
                )
                body = result.pop("response")
                record.update(result)
                record["model_response"] = extract_text(body)
                record["response"] = body
                record["usage"] = body.get("usage")
                request = OwaspLlm06VerifyRequest.model_validate(
                    {**row, "response": body, "tool_trace": record["tool_trace"]}
                )
                verdict = await verifier.verify(request)
                record["metrics"] = verdict.metrics
                record["approval_applicable"] = bool(request.verifier_metadata.policy.approval_required)
                if verdict.completion_judgment is not None:
                    record["completion_judgment"] = verdict.completion_judgment
            else:
                body = await gateway.send(payload)
                request = OwaspLlm02VerifyRequest.model_validate({**row, "response": body})
                verdict = await verifier.verify(request)
            record["model_response"] = extract_text(body)
            record["usage"] = body.get("usage")
            if record.get("turn_usage"):
                record["usage"] = {
                    key: sum(u.get(key, 0) for u in record["turn_usage"])
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                }
            record["reward"] = float(verdict.reward) if verdict.reward is not None else None
            if args.risk == "llm06":
                record["rollout_completed"] = True
        except Exception as exc:
            from nemo_gym.judge import JudgeError

            record["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, JudgeError):
                record["error_kind"] = "judge_error"
                from resources_servers.owasp_llm06.mock_tools import CaseSpec, evaluate

                spec = CaseSpec.model_validate(row["verifier_metadata"])
                record["metrics"] = evaluate(spec, record.get("tool_trace", []), answer=record["model_response"] or "")
                record["metrics"].pop("reward")
                record["approval_applicable"] = bool(spec.policy.approval_required)
            if hasattr(exc, "tool_trace"):
                record["tool_trace"] = exc.tool_trace
                record["turn_usage"] = exc.turn_usage
                record["error_kind"] = exc.kind
                record["last_response"] = exc.last_response
        record["attempts"] = gateway.attempts
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        return record


def summary(records: list[dict], args: argparse.Namespace) -> dict:
    if args.risk == "llm10":
        from benchmarks.owasp_llm10.reporting import build_summary

        return build_summary(records)
    if args.risk == "llm06":
        from benchmarks.owasp_llm06.reporting import build_summary

        return build_summary(records)
    groups = {}
    for row in records:
        groups.setdefault(row.get("source_dataset") or "unknown", []).append(row)

    def stats(items):
        good = [x for x in items if x["reward"] is not None]
        return {
            "total": len(items),
            "scored": len(good),
            "errors": len(items) - len(good),
            "mean_reward": round(sum(x["reward"] for x in good) / len(good), 5) if good else None,
            "zero_reward_cases": sum(x["reward"] == 0 for x in good),
        }

    return {
        "model": args.model,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "overall": stats(records),
        "by_dataset": {k: stats(v) for k, v in sorted(groups.items())},
        "warning": "Mean verifier reward is NOT a leakage rate; empty responses and password_gate task checks also affect reward.",
    }


async def main(
    args: argparse.Namespace,
    *,
    verifier: "OwaspLlm02Verifier | OwaspLlm06Verifier | StandaloneVerifier | OwaspLlm10Verifier | None" = None,
) -> None:
    if verifier is None and args.risk == "llm06" and getattr(args, "judge_config", None):
        from resources_servers.owasp_llm06.standalone import StandaloneVerifier

        verifier = StandaloneVerifier.from_config(
            Path(args.judge_config), env_file=getattr(args, "judge_env_file", None)
        )
        args.judge_profile = verifier.profile()
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError("Invalid shard index/count")
    rows = load_rows(args.input, args.shard_index, args.shard_count, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No rows for shard {args.shard_index}; writing empty result.")
    if args.risk in ("llm06", "llm10"):
        root = Path(__file__).resolve().parent
        try:
            git_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, errors="replace"
            )
            git_commit = git_result.stdout.strip() if git_result.returncode == 0 else None
        except FileNotFoundError:
            # Minimal evaluator images can run the benchmark without Git installed.
            git_commit = None
        metadata = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "parameters": {key: value for key, value in vars(args).items() if key != "api_key"},
            "git_commit": git_commit,
            "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                b"".join(
                    path.read_bytes()
                    for path in [
                        root / "run_standalone.py",
                        root / "benchmarks/response_utils.py",
                        *sorted((root / f"resources_servers/owasp_{args.risk}").glob("*.py")),
                        *(
                            [
                                root / "resources_servers/owasp_llm06/agent.py",
                                root / "resources_servers/owasp_llm06/mock_tools.py",
                                root / "benchmarks/owasp_llm10/reporting.py",
                                root / "aggregate.py",
                            ]
                            if args.risk == "llm10"
                            else []
                        ),
                    ]
                )
            ).hexdigest(),
        }
        metadata_path = out.with_suffix(".run.json")
        if args.risk == "llm10" and out.exists() and not metadata_path.exists():
            raise ValueError("Existing LLM10 results lack execution metadata; use a new run ID")
        if out.exists() and metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old.get("input_sha256") != metadata["input_sha256"] or any(
                old.get("parameters", {}).get(key) != getattr(args, key, None)
                for key in ("model", "risk", "shard_count", "shard_index", "judge_profile")
            ):
                raise ValueError("Existing run has different input/model/sharding; use a new run directory")
            if old.get("parameters", {}).get("completion_judge_model"):
                raise ValueError("Existing run used the former standalone judge; use a new run directory")
            if old.get("parameters", {}).get("gym_judge_profile"):
                raise ValueError("Existing run used the server-based judge; use a new run directory")
            if args.risk == "llm10" and (
                old.get("implementation_sha256") != metadata["implementation_sha256"]
                or any(
                    old.get("parameters", {}).get(k) != getattr(args, k, None)
                    for k in ("base_url", "max_output_tokens", "timeout", "run_id")
                )
            ):
                raise ValueError("Existing LLM10 run has different execution settings/code; use a new run ID")
        # Each shard stages its own temporary copy; atomic replacement avoids partial snapshots.
        snapshot_dir = out.parent.parent if out.parent.name == "shards" else out.parent
        for source, name in (
            (Path(args.input), "dataset.jsonl"),
            (Path(args.input).parent / "preparation_report.json", "preparation_report.json"),
        ):
            destination = snapshot_dir / name
            if (
                args.risk == "llm10"
                and name == "dataset.jsonl"
                and destination.exists()
                and (hashlib.sha256(destination.read_bytes()).hexdigest() != metadata["input_sha256"])
            ):
                raise ValueError("LLM10 run directory contains a different input snapshot; use a new run ID")
            if source.is_file() and not destination.exists():
                temporary = out.with_suffix("." + name + ".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
        out.with_suffix(".run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sem = asyncio.Semaphore(args.concurrency)
    if verifier is None and args.risk == "llm10":
        from resources_servers.owasp_llm10.app import OwaspLlm10Verifier

        verifier = OwaspLlm10Verifier()
    elif verifier is None and args.risk == "llm06":
        from resources_servers.owasp_llm06.app import OwaspLlm06Verifier

        verifier = OwaspLlm06Verifier()
    elif verifier is None:
        verifier = OwaspLlm02Verifier()
    # Load completed results from previous runs; never re-request scored cases.
    previous = {}
    if out.exists() and args.resume:
        with out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    previous[item["case_id"]] = item
                except (ValueError, KeyError):
                    pass
    pending = [
        r
        for r in rows
        if not (
            r.get("case_id") in previous
            and (
                previous[r["case_id"]].get("reward") is not None
                or (
                    previous[r["case_id"]].get("scoring_available") is False
                    and previous[r["case_id"]].get("rollout_completed")
                    and not previous[r["case_id"]].get("error")
                )
            )
        )
    ]
    print(f"Resuming: {len(rows) - len(pending)} completed already; {len(pending)} pending", flush=True)
    if args.risk == "llm06" and (pending or args.probe_only):
        from resources_servers.owasp_llm06.agent import probe_tools

        proof = {
            "model": args.model,
            "base_url": args.base_url,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "supported": False,
        }
        try:
            proof.update(await probe_tools(Gateway(args).send, args.model))
        except Exception as exc:
            proof["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            out.with_suffix(".capabilities.json").write_text(json.dumps(proof, indent=2) + "\n")
        if args.probe_only:
            print(json.dumps(proof, indent=2))
            return
    # Each task owns its request counters; all tasks share Gym's global aiohttp pool.
    if pending:
        tasks = [asyncio.create_task(process(row, verifier, args, sem)) for row in pending]
        # Append-only checkpoint log; flush and fsync each result so a killed pod
        # can resume. On restart the most recent record for each case wins.
        with out.open("a", encoding="utf-8") as f:
            for task in asyncio.as_completed(tasks):
                record = await task
                previous[record["case_id"]] = record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if args.fsync_every > 0 and (sum(t.done() for t in tasks) % args.fsync_every == 0):
                    os.fsync(f.fileno())
                done = len(rows) - len(pending) + sum(t.done() for t in tasks)
                if done % 25 == 0 or done == len(rows):
                    print(f"Shard {args.shard_index}: {done}/{len(rows)}", flush=True)
    records = [previous[r["case_id"]] for r in rows if r.get("case_id") in previous]
    # Compact checkpoints to one row per case, including unresolved errors.
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    report = summary(records, args)
    report_path = out.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.risk == "llm10" and args.shard_count == 1:
        from aggregate import aggregate

        aggregate(
            out.parent.parent if out.parent.name == "shards" else out.parent,
            shards=1,
            risk="llm10",
            expected_count=len(rows),
        )
    print(json.dumps(report, indent=2), flush=True)
    # Preserve persistent API failures as recorded errors; do not fail the entire shard.
    if report["overall"]["errors"]:
        print(f"WARNING: {report['overall']['errors']} unscored case(s); inspect {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--risk", choices=("llm02", "llm06", "llm10"), default="llm02")
    p.add_argument("--run-id", default="run-001")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--base-url", default=os.getenv("POLICY_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.getenv("POLICY_MODEL_NAME"))
    p.add_argument("--api-key", default=os.getenv("POLICY_API_KEY"))
    p.add_argument("--shard-index", type=int, default=int(os.getenv("JOB_COMPLETION_INDEX", "0")))
    p.add_argument("--shard-count", type=int, default=int(os.getenv("SHARD_COUNT", "1")))
    p.add_argument("--limit", type=int, default=0, help="Maximum cases per shard (0=all)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--fsync-every", type=int, default=25)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-tool-calls", type=int, default=32)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--judge-config", help="YAML config for the independent standalone judge")
    p.add_argument("--judge-env-file", default="env.yaml", help="Private YAML with independent judge settings")
    a = p.parse_args()
    a.model = a.model or ("sarvam-30b" if a.risk in ("llm06", "llm10") else "gpt")
    a.input = a.input or (
        f"benchmarks/owasp_{a.risk}/data/prepared.jsonl"
        if a.risk in ("llm06", "llm10")
        else "benchmarks/owasp_llm02/data/example.jsonl"
    )
    a.output = a.output or (
        f"results/llm10/{a.model}/{a.run_id}/shards/shard-{a.shard_index}.jsonl"
        if a.risk == "llm10"
        else f"results/llm06/{a.model}/run-003/shards/shard-{a.shard_index}.jsonl"
        if a.risk == "llm06"
        else "results/owasp_llm02/shard-0.jsonl"
    )
    if min(a.concurrency, a.timeout, a.max_turns, a.max_tool_calls) <= 0 or a.retries < 0 or a.limit < 0:
        p.error("Concurrency, timeout and agent limits must be positive; retries/limit must be nonnegative")
    if not a.api_key:
        p.error("POLICY_API_KEY is required (or pass --api-key)")
    if a.judge_config and a.risk != "llm06":
        p.error("--judge-config is supported for llm06")
    asyncio.run(main(a))
