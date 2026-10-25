# OWASP LLM02 — standalone Kubernetes batch evaluator

Uses the **unchanged** verifier in `resources_servers/owasp_llm02/app.py`, the prepared dataset `benchmarks/owasp_llm02/data/example.jsonl`, and an OpenAI-compatible `/v1/responses` endpoint. **No NeMo Gym head server or Ray.** The script still imports NeMo Gym Python classes.

## Install files into your existing NeMo Gym repository

From `~/nemo_gym/Gym`, copy `run_standalone.py`, `aggregate.py`, and `Dockerfile.llm02` to the repository root; copy `k8s/*.yaml` into `k8s/`. Append `.dockerignore.additions.txt` to `.dockerignore` without removing existing rules.

## Test locally first

```bash
export POLICY_API_KEY='YOUR_KEY'
python run_standalone.py --limit 10 --shard-count 1 --output results/owasp_llm02/smoke.jsonl
```

The runner reads POLICY_API_KEY, POLICY_BASE_URL, and POLICY_MODEL_NAME from environment variables; it does not automatically load the NeMo Gym policy YAML. The runner validates the raw `/v1/responses` JSON using `OwaspLlm02VerifyRequest.model_validate()`. If your model gateway produces a nonstandard Responses payload, the error is recorded per case and the process exits nonzero; fix the adapter instead of silently scoring as safe.

## Build and push

```bash
docker build -f Dockerfile.llm02 -t YOUR_REGISTRY/owasp-llm02:v1 .
docker push YOUR_REGISTRY/owasp-llm02:v1
```

Edit the `image:` field in both YAMLs to the pushed image.

## Kubernetes prerequisites

- Namespace `llm-logging` exists.
- PVC `owasp-results-pvc` exists and supports concurrent mounting **across nodes** (normally ReadWriteMany). Do not use a single-node RWO volume for pods scheduled on different nodes.
- Kubernetes pods can reach `http://172.17.99.1:31184/v1` (headnode reachability does not guarantee pod reachability).
- The dataset is included in the image. Rebuild image when dataset or verifier changes.
- Create a Kubernetes Secret without putting the API key in YAML or shell history:

```bash
read -rsp 'API key: ' API_KEY; echo
kubectl -n llm-logging create secret generic llm02-api-key --from-literal=api-key="$API_KEY"
unset API_KEY
```

## Execute

```bash
kubectl apply -f k8s/job.yaml
kubectl -n llm-logging get job owasp-llm02-eval -w
kubectl -n llm-logging get pods -l job-name=owasp-llm02-eval
```

After all 8 completions **succeed**:

```bash
kubectl apply -f k8s/aggregate-job.yaml
kubectl -n llm-logging logs job/owasp-llm02-aggregate
```

Final files on the PVC: `/results/llm02/all_results.jsonl` and `/results/llm02/summary.json`.

## Caveats

- Job name and result directory are fixed for this run. Use a fresh name and output directory for each new experiment to avoid mixing stale shards.
- Retryable API errors are retried within a pod. Failed pods can be restarted by Kubernetes; output is atomically replaced only after a complete shard finishes. If any case remains unscored, the pod exits nonzero and the aggregation job refuses to produce a complete report.
- A partial `*.tmp` file may remain after termination and can be safely ignored.
- Model responses can contain protected secrets. Secure the PVC, restrict access, and avoid sharing raw results publicly.
- `reward` is the **existing verifier's** score, not a direct privacy leakage rate. This verifier does not fully measure semantic disclosure or legitimate task completion.
- No additional judge model is required for the existing deterministic verifier.
