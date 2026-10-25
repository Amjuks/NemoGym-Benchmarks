#!/usr/bin/env python3
"""Kubernetes Indexed Job runner reusing the OWASP LLM02 NeMo Gym verifier."""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'resources_servers' / 'owasp_llm02'))
from app import OwaspLlm02Verifier, OwaspLlm02VerifyRequest  # noqa: E402


def extract_text(response):
    return '\n'.join(
        part.get('text', '')
        for item in response.get('output', [])
        if item.get('type') == 'message' and item.get('role') == 'assistant'
        for part in item.get('content', [])
        if part.get('type') == 'output_text'
    )


def load_rows(path, shard, shards, limit):
    selected = []
    with open(path, encoding='utf-8') as f:
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


async def process(row, client, verifier, args, sem):
    async with sem:
        started = time.monotonic()
        record = {
            'case_id': row.get('case_id'),
            'source_dataset': row.get('source_dataset'),
            'scenario_type': row.get('scenario_type'),
            'authorized': row.get('authorized'),
            'model': args.model,
            'shard_index': args.shard_index,
            'reward': None,
            'model_response': None,
            'usage': None,
            'error': None,
        }
        try:
            payload = {
                'model': args.model,
                'input': row['responses_create_params']['input'],
                'max_output_tokens': args.max_output_tokens,
                'store': False,
            }
            for attempt in range(args.retries + 1):
                try:
                    response = await client.post('responses', json=payload)
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        response.raise_for_status()
                    response.raise_for_status()
                    body = response.json()
                    if body.get('error'):
                        raise RuntimeError(str(body['error']))
                    break
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == args.retries:
                        raise
                    await asyncio.sleep(min(30, 2 ** attempt + random.random()))
            record['model_response'] = extract_text(body)
            record['usage'] = body.get('usage')
            # Pass the original Responses API object to the native Pydantic verifier request.
            request = OwaspLlm02VerifyRequest.model_validate({**row, 'response': body})
            verdict = await verifier.verify(request)
            record['reward'] = float(verdict.reward)
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['duration_seconds'] = round(time.monotonic() - started, 3)
        return record


def summary(records, args):
    groups = {}
    for row in records:
        groups.setdefault(row.get('source_dataset') or 'unknown', []).append(row)

    def stats(items):
        good = [x for x in items if x['reward'] is not None]
        return {
            'total': len(items), 'scored': len(good),
            'errors': len(items) - len(good),
            'mean_reward': round(sum(x['reward'] for x in good) / len(good), 5) if good else None,
            'zero_reward_cases': sum(x['reward'] == 0 for x in good),
        }
    return {
        'model': args.model, 'shard_index': args.shard_index, 'shard_count': args.shard_count,
        'overall': stats(records),
        'by_dataset': {k: stats(v) for k, v in sorted(groups.items())},
        'warning': 'Mean verifier reward is NOT a leakage rate; empty responses and password_gate task checks also affect reward.',
    }


async def main(args):
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError('Invalid shard index/count')
    rows = load_rows(args.input, args.shard_index, args.shard_count, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f'No rows for shard {args.shard_index}; writing empty result.')
    sem = asyncio.Semaphore(args.concurrency)
    verifier = OwaspLlm02Verifier()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip('/') + '/',
        headers={'Authorization': f'Bearer {args.api_key}'},
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        tasks = [asyncio.create_task(process(row, client, verifier, args, sem)) for row in rows]
        records = []
        # Write to a temporary path, atomically replace the final shard file on completion.
        tmp = out.with_name(out.name + '.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            for task in asyncio.as_completed(tasks):
                record = await task
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                if len(records) % 25 == 0 or len(records) == len(rows):
                    print(f'Shard {args.shard_index}: {len(records)}/{len(rows)}', flush=True)
        os.replace(tmp, out)
    report = summary(records, args)
    report_path = out.with_suffix('.summary.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    # Fail the job when scoring failed, so Kubernetes does not mark incomplete evaluation successful.
    if report['overall']['errors']:
        raise SystemExit(f"{report['overall']['errors']} case(s) failed; inspect {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='benchmarks/owasp_llm02/data/example.jsonl')
    p.add_argument('--output', default='results/owasp_llm02/shard-0.jsonl')
    p.add_argument('--base-url', default=os.getenv('POLICY_BASE_URL', 'http://172.17.99.1:31184/v1'))
    p.add_argument('--model', default=os.getenv('POLICY_MODEL_NAME', 'gpt'))
    p.add_argument('--api-key', default=os.getenv('POLICY_API_KEY'))
    p.add_argument('--shard-index', type=int, default=int(os.getenv('JOB_COMPLETION_INDEX', '0')))
    p.add_argument('--shard-count', type=int, default=int(os.getenv('SHARD_COUNT', '1')))
    p.add_argument('--limit', type=int, default=0, help='Maximum cases per shard (0=all)')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--max-output-tokens', type=int, default=1024)
    a = p.parse_args()
    if not a.api_key:
        p.error('POLICY_API_KEY is required (or pass --api-key)')
    asyncio.run(main(a))
