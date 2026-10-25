#!/usr/bin/env python3
"""Merge exactly one completed output per indexed shard and compute a global report."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--results-dir', default='/results/llm02')
p.add_argument('--shards', type=int, default=8)
a = p.parse_args()
folder = Path(a.results_dir)
records = []
seen = set()
for index in range(a.shards):
    path = folder / f'shard-{index}.jsonl'
    if not path.is_file():
        raise SystemExit(f'Missing completed shard: {path}')
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get('case_id')
        if key in seen:
            raise SystemExit(f'Duplicate case_id: {key}')
        seen.add(key)
        records.append(row)
if any(r['reward'] is None for r in records):
    raise SystemExit('Unscored cases exist; refusing to publish a complete report')
records.sort(key=lambda x: str(x['case_id']))
combined = folder / 'all_results.jsonl'
with combined.open('w', encoding='utf-8') as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def stats(items):
    return {
        'total': len(items),
        'mean_reward': round(sum(r['reward'] for r in items) / len(items), 5) if items else None,
        'zero_reward_cases': sum(r['reward'] == 0 for r in items),
    }
groups = defaultdict(list)
for row in records:
    groups[row.get('source_dataset') or 'unknown'].append(row)
report = {
    'overall': stats(records),
    'by_dataset': {k: stats(v) for k, v in sorted(groups.items())},
    'warning': 'Verifier reward is not synonymous with leakage rate.',
}
(folder / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
