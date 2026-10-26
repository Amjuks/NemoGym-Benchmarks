# LLM06 validation — 2026-10-25

Validation ran locally in `/home/aman/nemo_gym/Gym`. No full Kubernetes evaluation was launched.

| Check | Result |
|---|---|
| LLM06 tests plus existing LLM02 verifier fixture | 22 passed |
| Ten-case mock HTTP model through the shared runner | 10 scored, no errors, reward 1.0; two shards, resume made no additional requests |
| Every prepared row against native request and case schemas | 226 validated |
| Pinned source checksums and conversion reproducibility | All four prepared artifacts reproduced byte-for-byte |
| Ruff check and format check on changed Python | Passed |
| YAML parsing and shell syntax for both job commands | Passed |
| `kubectl apply --dry-run=client -f k8s/job-llm06.yaml` | Passed |
| Real Sarvam Responses API capability | Actual function call, tool-result acceptance, and marker acknowledgement observed |
| Real Sarvam LLM06 smoke | 10 scored, zero remaining errors, mean reward 0.50 |
| Real Sarvam LLM02 regression | One case scored, zero API/verifier errors; model reward 0.0 |

Tests:

```bash
.venv/bin/pytest resources_servers/owasp_llm06/tests resources_servers/owasp_llm02/tests -q
```

The live LLM06 smoke artifacts are under `results/llm06/sarvam-30b/smoke-001/`: `all_results.jsonl`, `summary.json`, `report.md`, and `shards/` with raw traces, capability evidence, and run metadata. Outputs are intentionally ignored by Git.

The first live pass scored nine cases with one output-budget truncation. Resume preserved those nine scored cases and retried the remaining case with a 4,096-token ceiling; the other cases used 2,048. Consequently this is a mixed-budget debugging smoke, not a uniform-parameter model comparison. The cluster manifest uses 4,096 for every case. Earlier probe attempts exposed the gateway's rejection of named-object tool choice and model variability in marker acknowledgement; capability testing uses `required` with a single function and reports acknowledgement separately.

Observed supported-control violations were 0/10. This does not establish that all repairs were safe: semantic reconciliation completion and unnecessary-sensitive-action ground truth are unavailable. Workflow completion was 8/10, exact file completion 1/2, and three malformed calls contributed zero rewards. Approval violation rate is unavailable because accepted source cases contain no trusted approval grants.

Repository-wide pre-commit was not claimed. This checkout is tracked from its parent directory, and the direct hook invocation rejects the existing `OWASP_README.md` underscore name required by this task. Changed source/config files were also checked in an isolated checkout with the repository's Python configuration: formatting, lint, whitespace, EOF, and verifier-config hooks passed. The environment-list update hook was excluded to avoid unrelated generated README changes; the private guardword catalog was unavailable. No commits were created.

## Minimal-image startup regression

A subsequent user-launched cluster job failed before any case ran because `python:3.13-slim` has no `git` executable. Git commit discovery now tolerates its absence and records `git_commit: null`, while retaining both content hashes. The ten-case mock HTTP smoke test now simulates missing Git throughout sharding, resume, and reporting. The existing LLM02 verifier fixture is included in the regression run. The failed full job was inspected but was not automatically restarted.
