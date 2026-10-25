# 02 — Developer Quickstart

## Prerequisites
- Access to the SoketLab headnode and Kubernetes namespace `llm-logging`.
- Python environment with the Gym checkout installed.
- Permission to read the model API key from the existing Kubernetes secret (do **not** copy its value into docs).
- Shared/pod-accessible code and results path. The current deployment used `hostPath`; **verify worker-node visibility** before relying on it.

## Inspect the project
```bash
cd ~/nemo_gym/Gym
git status --short
git rev-parse HEAD
python run_standalone.py --help
python aggregate.py --help
kubectl -n llm-logging get jobs
```

## Local smoke test
Export the appropriate `POLICY_BASE_URL`, `POLICY_API_KEY`, and `POLICY_MODEL_NAME` via your approved environment/secret mechanism. Then:
```bash
python run_standalone.py \
  --input benchmarks/owasp_llm02/data/example.jsonl \
  --output results/owasp_llm02/smoke.jsonl \
  --model sarvam-30b \
  --limit 10 --concurrency 2
```
Inspect `scored`, `errors` and individual case records. A low reward is not automatically an API failure.

## Full Kubernetes run
Edit `k8s/job.yaml` **before** applying:
- `POLICY_MODEL_NAME: sarvam-30b`
- `completionMode: Indexed`, `completions: 8`, `parallelism: 2`
- `--concurrency 2`
- a **new** output directory, e.g. `results/llm02/sarvam-30b/run-003/shards/`.
- the existing pod shell command must expand `${JOB_COMPLETION_INDEX}`; **do not run that pod command directly on the headnode**.

```bash
kubectl -n llm-logging delete job aman-owasp-llm02 --wait=true
kubectl apply -f k8s/job.yaml
kubectl -n llm-logging get job aman-owasp-llm02 -w
```
Delete only a job you own; ensure old outputs are archived before reusing a job name.

## Aggregate
The current aggregator expects `shard-0.jsonl` through `shard-7.jsonl` directly in `--results-dir`:
```bash
mkdir -p results/llm02/sarvam-30b/run-003/aggregate-input
cp results/llm02/sarvam-30b/run-003/shards/shard-*.jsonl \
   results/llm02/sarvam-30b/run-003/aggregate-input/
python aggregate.py \
  --results-dir results/llm02/sarvam-30b/run-003/aggregate-input \
  --shards 8
```
If the pod-owned directory is unwritable, aggregate in a **user-owned** folder. `aggregate.py` currently writes `all_results.jsonl` in the supplied directory. Confirm other output filenames by inspecting the script.

## Release gate
Never publish a score unless expected counts, successful scoring count, API error count and verifier semantics have all been checked.
