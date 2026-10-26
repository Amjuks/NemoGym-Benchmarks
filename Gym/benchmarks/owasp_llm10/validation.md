# LLM10 validation — 26 October 2026

- Prepared **3,000** distinct cases: 500 per category; 1,500 Alpaca train, 750 Instruct-v3 train,
  and 750 Instruct-v3 published test (explicit validation partition alias). Smoke: 20 cases,
  10 per dataset, with all six categories and two expected context rejections.
- Rebuilding from the pinned revisions and default seed reproduced byte-identical prepared JSONL,
  smoke JSONL, and preparation report. Prepared SHA-256:
  `ccffa5c38f24d711dbfcc9bcec7eb157dba213d589b9e22bd4b64163cd7dcb94`.
- `python -m pytest resources_servers/owasp_llm10/tests resources_servers/owasp_llm06/tests resources_servers/owasp_llm02/tests -q`:
  **116 passed** (57 LLM10 tests; 59 existing regression tests).
- Focused Ruff lint/format checks, dataset schema/provenance validation, `git diff --check`,
  CLI help checks, and Kubernetes YAML parsing passed. The Kubernetes Job was **not submitted**.

## Real model smoke

Used the existing private policy configuration with `sarvam-30b`, concurrency 2, 512-token
provider ceiling, 60-second timeout, and no transport retries:

```bash
.venv/bin/python run_standalone.py --risk llm10 --model sarvam-30b \
  --run-id smoke-20261026-01 --input benchmarks/owasp_llm10/data/smoke.jsonl \
  --limit 20 --concurrency 2 --retries 0 --timeout 60 --max-output-tokens 512
```

[Generated Markdown report](../../results/llm10/sarvam-30b/smoke-20261026-01/report.md)
and JSON/JSONL artifacts are under the same run directory.

| Measurement | Observed |
| --- | --- |
| Scored cases / total | 20/20 |
| Infrastructure/evaluation errors / total | 0/20 |
| Reward-1 cases / scored | 6/20 |
| Budget-compliant cases / measurable | 20/20 |
| Actual token/request/tool/time/step breaches / measurable (each type) | 0/20 |
| Task completion / objectively checkable model tasks | 5/18 |
| Normal-control completion / normal controls | 0/3 |
| Provider output stops / total | 13/20 |
| Input admission stops / total | 3/20 |
| Completed final responses / total | 4/20 |

All recorded network-attempt counts matched model-request telemetry; agent steps reconciled
with requests plus executed tools. Five synthetic lookup calls succeeded, with real tool-result
round trips and no redundant successful reads. Two input stops were expected oversized contexts;
a repetition case completed its useful lookup but its accumulated estimated context blocked the
next inference, so it failed that scenario's expected behavior. Models were never forced to loop.

The three normal controls spent their reported output allowance on reasoning (510 reported output
and reasoning tokens per case), returned no final answer, and scored zero. Thirteen provider-capped
cases are scored policy/task outcomes, not omitted evaluations. **100% budget compliance is not
100% task success.** No real request/tool/step quota stop was needed in this sample; controlled unit
rollouts cover those boundaries. This smoke is not evidence about every policy or source task.

Input admission uses labeled conservative byte reservations, not an exact provider tokenizer;
semantic quality of original source instructions is not judged. Source-derived data remains subject
to its source licenses and is excluded from Apache source releases. Review normal-control reasoning
allocation and conservative context reservations before comparing task quality under different
settings; use a fresh run ID and preserve the original report when changing a policy.
