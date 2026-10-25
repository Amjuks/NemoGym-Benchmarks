# 09 — Engineering Standards and Pull Request Checklist

## Required conventions
- Risk folder: `owasp_llmXX`; human label: `LLMXX:2025`.
- One benchmark and verifier per risk; shared utilities only for proven repeated needs.
- Source dataset revision and license recorded.
- Prepared JSONL: one JSON object per line, stable case ID, model input in `responses_create_params.input`, verifier-only truth separated.
- New verifier logic covered by positive, negative, authorized, unauthorized, malformed and judge-error tests.
- No real side-effecting tools in LLM06; mock-only sandbox.
- Bounded requests, tokens, wall-time and tools in LLM10.
- Model configuration and secrets outside source code.
- Each run has immutable `risk/model/run-id` output directory.
- All reported metrics state numerator, denominator and exclusions.
- Never equate mean reward to leakage rate.

## Before opening a PR
- [ ] `git diff` contains no secrets, personal data, generated outputs or huge model files.
- [ ] Docs list source revision, transformations and excluded rows.
- [ ] All converted records parse and match the verifier's typed request.
- [ ] Verifier tests pass, including adversarial controls.
- [ ] 10-case smoke run has nonzero scored cases and acceptable error count.
- [ ] Kubernetes job uses new output run ID and valid permissions/storage.
- [ ] Summary has expected total, scored, errors and source breakdown.
- [ ] Reporting examples are redacted.
- [ ] Changes to reward/judge are versioned and explained.
- [ ] README/handbook updated.

## Review responsibility
Dataset owner validates provenance and labels; verifier owner validates scoring; infra owner validates sandbox and resource caps; reviewer checks reproducibility and reports.

## Git
Keep the upstream Gym checkout as a separate repository unless you intentionally decide to manage it as a submodule. A parent `.gitignore` cannot override tracked files inside `Gym/.git`.
