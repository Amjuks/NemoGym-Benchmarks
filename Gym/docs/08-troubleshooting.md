# 08 — Troubleshooting

| Symptom | Cause / next action |
|---|---|
| `--shard-index: invalid int value: ''` | `${JOB_COMPLETION_INDEX}` was expanded on headnode; execute it inside Indexed Job pod. |
| `Missing completed shard: shard-0.jsonl` | Wrong results directory, incomplete shard output or files not visible. Check `find results -name 'shard-*.jsonl'`. |
| `PermissionError: ... all_results.jsonl` | Pod wrote directory as root. Aggregate copied shards in a user-owned directory or fix UID/GID/ACLs. |
| Job `Complete 8/8` but `scored: 0` | Runner recorded errors and exited normally. Inspect case-level `error` fields. |
| HTTP 503 `no_available_workers` | LiteLLM backend workers unhealthy/circuits open; stop load and check backend health with cluster operator. |
| HTTP 400 `stream ... input_value=None` | Backend rejected `stream: null`; test explicit `stream: false`, compare Responses vs Chat Completions translation. |
| `RemoteProtocolError` | Transient connection failure; retry with backoff, record unresolved errors. |
| All rewards 0 or very low | Inspect verifier outcome, not just reward. Zero reward is not automatically leakage. |
| Model outputs absent | Check gateway route, credentials, response parsing and worker health. |
| Duplicate/missing cases | Check shard count, stable IDs, resume logic and aggregation deduplication. |

## Diagnose case errors
```bash
python - <<'PY'
import json
from collections import Counter
p='results/llm02/sarvam-30b/run-003/aggregate-input/all_results.jsonl'
errors=Counter()
with open(p, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            row=json.loads(line)
            if row.get('error'):
                errors[str(row['error'])[:500]] += 1
for msg,count in errors.most_common(10):
    print(count, msg)
PY
```
Adapt the path to your run. Never paste raw secrets from logs into public tickets.

## Escalation
Include job name, pod names, model alias, UTC timestamps, error counts, sanitized error examples, Git SHA and dataset hash. Do not share API keys.
