# 06 — Kubernetes Jobs and Cluster Execution

## Current pattern
Namespace: `llm-logging`; job: `aman-owasp-llm02`; Indexed Job with 8 completions. Working directory inside pod: `/pipeline`. Model endpoint was the internal LiteLLM service:
`http://litellm-proxy.llm-logging.svc.cluster.local:4000/v1`.

The job manifest references Kubernetes secret `litellm-secret`, key `LITELLM_MASTER_KEY`. **Never commit the secret's value.**

## Pod command (illustrative; embed in job.yaml)
```yaml
command:
  - /bin/sh
  - -c
  - |
    set -e
    cd /pipeline
    pip install -q -e .
    python run_standalone.py \
      --input benchmarks/owasp_llm02/data/example.jsonl \
      --output "/pipeline/results/llm02/sarvam-30b/run-003/shards/shard-${JOB_COMPLETION_INDEX}.jsonl" \
      --shard-index "${JOB_COMPLETION_INDEX}" \
      --shard-count 8 \
      --concurrency 2 \
      --timeout 120 \
      --limit 0
```
`JOB_COMPLETION_INDEX` is set **inside Indexed Job pods**. Running this command on the headnode produces an empty shard index.

## Preflight
```bash
kubectl -n llm-logging get job aman-owasp-llm02
kubectl -n llm-logging get pods -l app=owasp-llm02
kubectl -n llm-logging get secret litellm-secret
```
Ensure all pods can read the same code/dataset and write to a shared result path. The earlier `hostPath: /home/aman/nemo_gym/Gym` arrangement is **not guaranteed shared across nodes**. Prefer an approved PVC/network share where available.

## Launch and monitor
```bash
kubectl -n llm-logging delete job aman-owasp-llm02 --wait=true
kubectl apply -f k8s/job.yaml
kubectl -n llm-logging get job aman-owasp-llm02 -w
kubectl -n llm-logging logs -l app=owasp-llm02 --all-containers=true --prefix=true --tail=30
```
Pod completion only means the process exited successfully; **it does not mean the cases were scored**. Inspect error counts.

## Permissions
The pods may write as root, preventing the headnode user from aggregating in place. Prefer a pod security context with the **actual** UID/GID (`id -u`, `id -g`) and compatible volume permissions; alternatively aggregate copies in a user-owned directory. Do not blindly `chown` shared cluster paths.

## Safe load
Start with `parallelism: 2`, `--concurrency 2` and a 10-case smoke test. Increase only after confirming inference worker health. A 503 `no_available_workers` means the backend needs attention, not more requests.

## LLM06 indexed job

`k8s/job-llm06.yaml` uses the existing secret and LiteLLM service, eight indexed shards and eight pods. It pins the hostPath checkout to `soketlab-worker001`. Each shard checks tool support before cases run. The full job is intentionally manual:

```bash
kubectl apply -f k8s/namespace-owasp.yaml
.venv/bin/python k8s/setup-owasp-secrets.py --env-file env.yaml
kubectl apply -f k8s/job-llm06.yaml
kubectl wait -n aman-owasp-bench --for=condition=complete job/aman-owasp-llm06-v3-gpt-judge --timeout=24h && \
.venv/bin/python aggregate.py --risk llm06 --shards 8 \
  --input benchmarks/owasp_llm06/data/prepared.jsonl \
  --results-dir results/llm06/sarvam-30b/run-004-gpt-judge
```

See [LLM06](../benchmarks/owasp_llm06/README.md) for preparation and smoke commands. Do not reuse a run directory with different input ordering, shard count, model, or verifier version. LLM02's `k8s/job.yaml` remains independent.

Version 3 includes all 4,220 source rows, with scoring unavailable explicitly recorded for rows without reviewed adapters. It uses a distinct job name and initializes its generated output directories with the checkout owner. Runs snapshot their inputs for aggregation. Root-owned historical runs can be aggregated with `--output-dir` to a writable location.

The LLM06 manifest uses an in-process standalone `gpt` judge client with independently supplied `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL_NAME`. Its private runtime YAML is separate from policy configuration. No Gym head, resources, or model servers are launched. The new job/run name prevents mixing judge-scored outputs into prior workflow-only runs.

LLM06 jobs and their `aman-owasp-api-keys` secret now live in the dedicated `aman-owasp-bench` namespace. The bootstrap script reads only independently defined local `policy_api_key` and `judge_api_key` values; it does not copy a secret from another namespace. Evaluator pods stay pinned to worker001 for hostPath access, with up to 16 concurrent case rollouts. The existing inference service remains reachable by its fully qualified cross-namespace DNS name.

## LLM10 Indexed Job

Prepare `benchmarks/owasp_llm10/data/prepared.jsonl` on the worker001 checkout, then manually
create `k8s/llm10-job.yaml`. It reuses the existing LiteLLM policy secret, runs eight indexed
shards with parallelism two and concurrency two per pod, and disables inference retries.
`metadata.generateName` provides a fresh shared run ID on each `kubectl create`; pod retries
resume checkpoints of that Job. No LLM10 full evaluation has been launched automatically.
[Exact launch, wait, and aggregation commands](../benchmarks/owasp_llm10/README.md#kubernetes-manual-launch-only).
