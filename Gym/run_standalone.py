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



def normalize_usage(response):
    """Normalize OpenAI-compatible gateway usage for NeMo Gym's Responses schema."""
    usage = response.get('usage') or {}
    input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
    output_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
    prompt_details = usage.get('prompt_tokens_details') or {}
    completion_details = usage.get('completion_tokens_details') or {}
    input_details = usage.get('input_tokens_details') or {}
    output_details = usage.get('output_tokens_details') or {}
    response['usage'] = {
        **usage,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': usage.get('total_tokens', input_tokens + output_tokens),
        'input_tokens_details': {
            **input_details,
            'cached_tokens': input_details.get('cached_tokens', prompt_details.get('cached_tokens', 0)),
        },
        'output_tokens_details': {
            **output_details,
            'reasoning_tokens': output_details.get('reasoning_tokens', completion_details.get('reasoning_tokens', usage.get('reasoning_tokens', 0))),
        },
    }
    return response


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
                "stream": False,
            }
            record['attempts'] = 0
            for attempt in range(args.retries + 1):
                record['attempts'] += 1
                try:
                    response = await client.post('responses', json=payload)
                    if response.is_error:
                        # Capture the actual API rejection reason for 400 diagnostics.
                        detail = response.text[:1500]
                        retryable = response.status_code in (408, 429, 500, 502, 503, 504) or (
                            response.status_code == 400 and 'could not decode header' in detail.lower()
                        )
                        if retryable:
                            raise httpx.HTTPStatusError(
                                f'HTTP {response.status_code}: {detail}', request=response.request, response=response
                            )
                        raise RuntimeError(f'HTTP {response.status_code}: {detail}')
                    body = response.json()
                    if body.get('error'):
                        raise RuntimeError(str(body['error']))
                    break
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == args.retries:
                        raise
                    await asyncio.sleep(min(30, 2 ** attempt + random.random()))
            body = normalize_usage(body)
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
    # Load completed results from previous runs; never re-request scored cases.
    previous = {}
    if out.exists() and args.resume:
        with out.open(encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    previous[item['case_id']] = item
                except (ValueError, KeyError):
                    pass
    pending = [r for r in rows if not (r.get('case_id') in previous and previous[r['case_id']].get('reward') is not None)]
    print(f'Resuming: {len(rows)-len(pending)} scored already; {len(pending)} pending', flush=True)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip('/') + '/',
        headers={'Authorization': f'Bearer {args.api_key}'},
        timeout=httpx.Timeout(args.timeout),
        limits=httpx.Limits(max_connections=max(32, args.concurrency * 2), max_keepalive_connections=max(16, args.concurrency)),
    ) as client:
        tasks = [asyncio.create_task(process(row, client, verifier, args, sem)) for row in pending]
        # Append-only checkpoint log; flush and fsync each result so a killed pod
        # can resume. On restart the most recent record for each case wins.
        with out.open('a', encoding='utf-8') as f:
            for task in asyncio.as_completed(tasks):
                record = await task
                previous[record['case_id']] = record
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                if args.fsync_every > 0 and (sum(t.done() for t in tasks) % args.fsync_every == 0):
                    os.fsync(f.fileno())
                done = len(rows) - len(pending) + sum(t.done() for t in tasks)
                if done % 25 == 0 or done == len(rows):
                    print(f'Shard {args.shard_index}: {done}/{len(rows)}', flush=True)
    records = [previous[r['case_id']] for r in rows if r.get('case_id') in previous]
    # Compact checkpoints to one row per case, including unresolved errors.
    tmp = out.with_name(out.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    report = summary(records, args)
    report_path = out.with_suffix('.summary.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    # Preserve persistent API failures as recorded errors; do not fail the entire shard.
    if report['overall']['errors']:
        print(f"WARNING: {report['overall']['errors']} unscored case(s); inspect {out}", flush=True)


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
    p.add_argument('--retries', type=int, default=5)
    p.add_argument('--fsync-every', type=int, default=25)
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-output-tokens', type=int, default=1024)
    a = p.parse_args()
    if not a.api_key:
        p.error('POLICY_API_KEY is required (or pass --api-key)')
    asyncio.run(main(a))
