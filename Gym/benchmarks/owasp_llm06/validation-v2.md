# Version 2 validation — 2026-10-26

No full version 2 benchmark was launched. The prepared release has **587 cases** from the pinned 4,220 source rows: **546 augmented + 41 training**. Every input has a reasoned adapter or an explicit exclusion; selection has no sampling cap.

| Validation | Result |
|---|---|
| Current LLM06 tests plus LLM02 verifier fixture | **40 passed** |
| Native request/case schema validation | All 587 prepared rows validated |
| Source-to-mock operation check | 3,534 tool executions checked without runtime exceptions |
| Preparation reproducibility | Prepared, smoke, exclusions, and preparation-report artifacts reproduced byte-for-byte |
| Historical reward compatibility | All 225 originally scored traces replayed with identical rewards |
| Mock HTTP model | Ten cases across two shards; resume made no additional requests; approval-positive/negative controls passed |
| Optional judge validation | Strict output validation, uncertainty handling, failure retention, and inability to override deterministic safety covered |
| Live optional judge call | Sarvam returned `completed: false` for an explicit refusal with no tool actions |
| Live version 2 development smoke | Ten source cases scored, no execution errors; reward 0.20, with actual permission/approval violations recorded |
| Focused final access-state view smoke | Two cases scored, no execution errors; one approval violation and two permission violations recorded |
| Scoped lint/format/whitespace/verifier-config hooks | Passed in an isolated checkout using repository Python settings |
| Kubernetes version 2 manifest | YAML/shell syntax and client dry-run passed; job was not created |

Commands:

```bash
.venv/bin/pytest resources_servers/owasp_llm06/tests resources_servers/owasp_llm02/tests -q
.venv/bin/ruff check run_standalone.py aggregate.py benchmarks/owasp_llm06 resources_servers/owasp_llm06
.venv/bin/ruff format --check run_standalone.py aggregate.py benchmarks/owasp_llm06 resources_servers/owasp_llm06
kubectl apply --dry-run=client -f k8s/job-llm06.yaml
```

## Artifacts and interpretation

- `results/llm06/sarvam-30b/run-001-audit/` adds source coverage, reliability diagnostics, and explicit unscored-case reporting to the unchanged historical scores.
- `results/llm06/sarvam-30b/smoke-v2-002/` records a ten-case development snapshot; the final access-context view was subsequently validated in `results/llm06/sarvam-30b/access-v2-smoke/`. Each directory contains its actual immutable input snapshot and hashes. These small development runs are integration evidence, not a uniform full-release model comparison.
- `results/llm06/sarvam-30b/completion-judge-smoke.json` records the single live judge check. This validates the call path and output contract, not judge calibration or independence.
- The archived version 2 dataset is `benchmarks/owasp_llm06/data/v2/prepared.jsonl`; a full evaluation must use a fresh `run-002` as configured in the manifest. The runner refuses to mix changed input/model/shard/judge configurations into existing output files.

The source model made real failures in the new controls: some calls attempted access before required approval or used an area outside the source's requested area. These were denied and scored as attempts. No inference errors were reclassified as security violations. The empty-wrapper normalization is narrow and versioned; the recorded smoke trace did not exercise every possible wrapper. Unit controls cover both normalized and rejected forms.

## Recovery of the original incomplete case

The original `run-001` is preserved. Its single unscored case was retried in a **separate** directory, `run-001-recovered/`, using the archived version 1 input, original sharding/model, and version 1 grading behavior. The runner logged **28 already scored / 1 pending** for shard 0; only that pending case received a fresh rollout. Its new reward was 0, with no remaining execution error.

The recovered aggregate contains **226 scored / 226 total, zero errors**, mean reward **0.207965**. All **225** previously scored result objects are unchanged. The recovery used an 8,192-token ceiling for the one retried case while the original full run used 4,096; this is a disclosed mixed-budget recovery, not a new uniform baseline. Version 2 normalization was not applied to historical traces.

## Check limits

Repository-wide pre-commit is not claimed: the parent-repository layout and existing `OWASP_README.md` underscore hook remain as described in the earlier validation record. Scoped checks exclude the unrelated generated environment-list table update; the private guardword catalog is unavailable. LLM02's data and verifier remain unchanged. No commits were created.

There are still **3,633 excluded source rows**, mostly unreviewed training environments, and **552 workflow-proxy cases by default**. General semantic repair correctness and unnecessary-action truth cannot be fabricated. The optional judge can return unknown; it does not create missing source permissions. These limits are detailed in the implementation audit.
