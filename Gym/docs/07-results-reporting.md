# 07 — Results, Aggregation and Reporting Standard

## Single location
All future outputs should be under **`Gym/results/`**:
```text
results/
  llm02/
    sarvam-30b/
      run-003/
        shards/
          shard-0.jsonl ... shard-7.jsonl
        all_results.jsonl
        summary.json
        report.json          # planned
        report.md            # planned
        report.html          # planned
        run_manifest.json    # planned
  llm06/
  llm10/
reports/
  index.html                 # planned cross-run index
```
These are **target conventions**, not evidence that report generation already exists.

## Existing aggregation CLI
```bash
python aggregate.py --results-dir DIRECTORY_CONTAINING_SHARDS --shards 8
```
It requires `shard-0.jsonl` ... `shard-7.jsonl` directly in the directory and writes `all_results.jsonl` there. Its summary has keys:
`overall.total`, `overall.scored`, `overall.errors`, `overall.mean_reward`, `overall.zero_reward_cases`, and `by_dataset`. Verify the actual summary output filename in the installed script.

If shards are pod-owned:
```bash
mkdir -p ~/nemo_gym/Gym/results/llm02/sarvam-30b/run-003/aggregate-input
cp /PATH/TO/SHARDS/shard-*.jsonl \
   ~/nemo_gym/Gym/results/llm02/sarvam-30b/run-003/aggregate-input/
python aggregate.py \
  --results-dir ~/nemo_gym/Gym/results/llm02/sarvam-30b/run-003/aggregate-input \
  --shards 8
```
If the parent `results/` itself is not writable, use a user-owned staging directory and ask the storage administrator to correct ownership/ACLs.

## Proposed report contract
Every risk report must show:
1. **Run identity:** risk, model, run ID, timestamps, code/data/verifier versions.
2. **Execution health:** expected, attempted, scored, missing, API errors, verifier errors.
3. **Risk-specific metrics:** clearly named, with numerators and denominators.
4. **Task quality:** legitimate completion and over-refusal where applicable.
5. **Slices:** source dataset, scenario type, authorization, difficulty, language where present.
6. **Resources:** token totals, latency percentiles, throughput, estimated cost when known.
7. **Failure analysis:** reason codes, redacted examples, actionable mitigations.
8. **Limitations:** dataset bias, judge limitations, model/backend outages.

## Suggested run manifest (new proposed format)
```json
{
  "risk": "llm02",
  "model": "sarvam-30b",
  "run_id": "run-003",
  "expected_cases": 3232,
  "shards": 8,
  "git_commit": "<commit-sha>",
  "dataset_sha256": "<hash>",
  "verifier_version": "<version>",
  "started_at": "<ISO8601>",
  "model_endpoint_alias": "cluster-litellm",
  "execution_mode": "kubernetes-indexed-job"
}
```
The manifest is **proposed**, not an existing NeMo Gym schema.

## Mandatory metric interpretation
- `scored / attempted`: evaluation completion, **not** safety.
- `mean_reward`: task/verifier score, **not** leakage rate.
- `unauthorized_disclosures / eligible_unauthorized_cases`: disclosure rate **only if explicit disclosure verdicts exist**.
- `unauthorized_actions / applicable_agentic_cases`: LLM06 risk metric **only with executed mock-tool traces**.
- `budget_violations / applicable_cases`: LLM10 risk metric **only with measured usage**.
- Report API errors separately; never include them as safe model responses.

## Comparison policy
Compare runs only if dataset selection, verifier version, tool simulator, prompt and judge settings are compatible. Show differences when they are not.

## LLM06 JSON and Markdown

`aggregate.py --risk llm06 --results-dir DIR --shards N --input DATA.jsonl` reads either `DIR/shards/shard-N.jsonl` or the earlier flat shard layout. It requires every expected input ID and refuses duplicates before publishing `all_results.jsonl`, `summary.json`, and `report.md`.

Each security/completion rate stores numerator, eligible denominator, and nullable rate. Infrastructure and truncated trajectories remain unscored errors. Legitimate completion excludes reconciliation cases without semantic ground truth; workflow coverage is reported separately. Approval rates use only source-backed approval controls; unnecessary-sensitive-action rates remain unavailable without source truth. The Markdown report includes dataset/scenario breakdowns and representative scored failures by ID, without copying tool data or protected content.

Version 2 reporting includes source coverage/exclusions, tool-error breakdowns, completion methods and judge models, security-check pass rates, and unscored-case IDs. Resume rejects changed input/model/sharding/judge profiles. Use the original `data/v1/prepared.jsonl` for version 1 outputs, or let aggregation use a newer run's immutable input snapshot. The optional completion judge cannot override deterministic safety checks.

Version 3 retains every source row. Reports separate completed rollouts, scored cases, cases without scoring adapters, and API/trajectory errors. Unknown safety/completion metrics stay null and do not enter scored denominators. Resume skips successfully completed unscorable rollouts.

Standalone judge results preserve the verdict, raw judge response, model-server/model identity, and usage in `completion_judgment`. Independent judge endpoint/model/generation/prompt settings are stored in the credential-free run profile and checked on resume. Judge transport failures remain unscored, and the policy response is retained.

## LLM10 JSON and Markdown

`aggregate.py --risk llm10 --results-dir results/llm10/MODEL/RUN --shards 8` uses the immutable
input snapshot and checks all shard IDs and deterministic verdicts before writing
`all_results.jsonl`, `summary.json`, and `report.md`. Reports show explicit numerators and
eligible denominators for reward, resource compliance/violations, termination and task
completion, with dataset/scenario slices and normal-control performance. Actual consumption,
blocked attempts, redundant model calls, unavailable tokens, and infrastructure errors remain
separate. Token estimates never enter measured-token statistics. [Interpretation and limitations](../benchmarks/owasp_llm10/README.md).
