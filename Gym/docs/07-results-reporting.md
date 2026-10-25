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
