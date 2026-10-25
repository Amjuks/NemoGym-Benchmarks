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
