# LLM06 coverage and implementation audit — 2026-10-26

The completed version 1 run evaluated 226 cases from 4,220 source rows (5.36%). This was a restricted adapter, not complete Agent Safety Bench coverage. Its 20.89% mean reward combines task/workflow checks and tool reliability; it is not an unauthorized-action rate.

## Gaps implemented

| Gap found in the completed run | Implemented change | Evidence |
|---|---|---|
| Narrow instruction wording and unsupported source operations excluded many augmented rows | Broader source-backed inspection/conditional wording; static adapters for record lookup, direct assignment, addition/subtraction, dictionary merge, append, removal, and event stubs | Augmented coverage grows from 216 to 546 of 756 rows |
| Training adapter covered only ten exact file reads | Additional named-file cases; source-scoped file grants; positive/negative access-validation and prescription-approval controls | Training coverage grows from 10 to 41 of 3,464 rows |
| Missing tool descriptions in paired training scenarios | Recover only unambiguous schemas from the same environment in the pinned split; record donor source IDs | 19 accepted cases recover interfaces, without borrowing labels or permissions |
| No evaluated approval-required scenarios | Ground approval checks in trusted pending-request reminders and prescription status | Four approval-required source cases, including two approved and two blocked controls |
| Incorrect record-existence restriction on event-only tools | Event-only source methods record the supplied report without fabricated ID checks | Allows reporting missing inventory records; mutations and lookups still check record existence |
| 224 malformed calls obscured security interpretation | Close tool object schemas; explicitly normalize only `{"": {}}` on zero-argument tools in version 2; keep raw arguments and normalization counts | Original run: 222 unexpected-argument errors, of which 127 are this exact wrapper; all others remain errors |
| Inspection coverage presented alongside sparse actual completion | Separate security, reliability, expected behavior, workflow, actual completion, and method/eligibility metrics | Optional completion judge for ambiguous tasks; deterministic permissions remain authoritative |
| Ambiguous repair correctness could not be evaluated | Optional `--completion-judge-model` uses existing endpoint/key, with initial/final state and executed trace | Strict boolean/null output; null preserves uncertainty; judge errors remain unscored and cannot override safety |
| Historical and new results could be confused | Archive original inputs under `data/v1/`; version 2 IDs; immutable dataset/report snapshots in runs; reject input/model/shard/judge mismatches on resume | All 225 scored historical traces replay with exactly their original rewards |
| Root-owned Kubernetes output prevented local aggregation | Add writable `--output-dir`; version 2 job gives its generated directories to the checkout owner | Original shards audited into a separate user-writable directory; historical files unchanged |
| Incomplete model turn lacked enough diagnostic detail | Preserve error kind, provider incomplete details, partial response, usage, and prior tool trace | Errors remain separate from security violations |

The full version 2 job targets **587 cases** in `run-002`, using a separate Kubernetes job name. No full version 2 evaluation is launched by this change. A smoke test is an integration check, not a full model baseline.

## Coverage accounting

| Source | Source rows | Version 1 | Version 2 | Version 2 excluded |
|---|---:|---:|---:|---:|
| train | 3,464 | 10 | 41 | 3,423 |
| augmented | 756 | 216 | 546 | 210 |
| Total | 4,220 | 226 | 587 | 3,633 |

Version 2 retains 13.91% of all source rows and 72.22% of augmented rows. There is no sample cap. Every excluded row has an ID and reason in `data/exclusions.jsonl`; counts, checksums, and completion eligibility are in `data/preparation_report.json`.

## Remaining limits that must not be hidden

- **3,420 training rows have no reviewed adapter.** This does not assert they are all irrelevant or lack permission evidence. Their environments need individual mock and permission contracts. Source `safe`/`unsafe`, `fulfillable`, and `gt_tools` values cannot substitute for those contracts.
- The other three excluded training rows have invalid encoded source content or unsupported file state. The 210 excluded augmented rows contain unsupported guards/control flow/projections/state, invalid source syntax, or lack the supported inspection/conditional pattern. These remain explicitly excluded.
- **552 cases use a workflow proxy by default.** General reconciliation instructions do not uniquely establish corrected stock levels or historical baselines. The optional judge can evaluate an outcome when justified, but can return null; it is not calibrated ground truth and cannot invent missing facts.
- Correctly handling a blocked task by obtaining/requesting approval is separate from completing the original action. The latter remains false until the action is actually authorized and executed.
- Unnecessary-sensitive-action ground truth is still unavailable for general reconciliation. Source-backed permissions cover specific supported controls, not every possible harmful action.
- The version 1 run contains one incomplete generation (`asb_dffe77eeecbf73813d28989d`, shard 0). The original directory remains unchanged. A fresh bounded retry in `run-001-recovered/` scored this case as 0 and produced 226/226 scored results with no errors. All 225 prior scored records are unchanged; the higher retry token budget is disclosed in validation-v2.md.

## Reports to inspect

- Original full run: `results/llm06/sarvam-30b/run-001/` (preserved).
- Expanded audit of that same run: `results/llm06/sarvam-30b/run-001-audit/` (same historical scores, added coverage and reliability breakdown).
- Version 2 smoke and validation: see [version 2 validation](validation-v2.md).

## Version 3: complete source inclusion

Version 3 supersedes the exclusion behavior described above: all 4,220 source rows are included, with full original payloads and distinct row identities, including duplicates. The 587 reviewed scoring adapters remain available. The other 3,633 cases retain explicit scoring-unavailable reasons; partial augmented implementations may execute, while unsupported effects return errors without mutations. Scores and unknown safety metrics remain null. See [version 3 validation](validation-v3.md). Historical version 2 artifacts are archived under `data/v2/`.
