# 01 — Current System and Verified Scope

## Confirmed from the evaluation workflow
- Headnode project: `/home/aman/nemo_gym/Gym`.
- OWASP risk implemented: **LLM02:2025 — Sensitive Information Disclosure**.
- Dataset: `benchmarks/owasp_llm02/data/example.jsonl`, **3,232 records**:
  - `password_eval`: 1,000
  - `peep`: 1,984
  - `privawarebench`: 248
- Additional sampling artifact reported: `benchmarks/owasp_llm02/data/balanced_90.jsonl`.
- Verifier entry point: `resources_servers/owasp_llm02/app.py`, with `OwaspLlm02Verifier` and `OwaspLlm02VerifyRequest`.
- Execution: `run_standalone.py`, OpenAI-compatible Responses API, Kubernetes Indexed Jobs.
- Aggregation: `aggregate.py --results-dir DIR --shards N`.
- Model tested: `sarvam-30b` via LiteLLM; a ten-case smoke test scored 10/10, with mean reward 0.1. A prior 3,232-case run produced 3,232 infrastructure/API errors; a later `-v2` run was reported as looking good, **but its numerical summary was not provided**. Do not claim a measured leakage rate.

## Not yet verified or implemented
- Native Gym end-to-end execution replacing the standalone runner.
- Fully implemented standardized Markdown/HTML/JSON reporting layer.
- LLM10 full Kubernetes evaluation (the benchmark and bounded runner are implemented; see below).
- Central `results/llmXX/model/run-id/` convention in all jobs.
- Exact current contents of remote scripts, Kubernetes manifests and verifier request/response schemas.

## Model reward caveat
`mean_reward` is **not** a sensitive-information leakage rate. Zero reward can arise from password-gate task checks, empty responses and other verifier logic. A report must distinguish:
1. Model/API execution failure.
2. Verifier/judge failure.
3. Verified unauthorized disclosure.
4. Legitimate task failure or unnecessary refusal.
5. Successful authorized behavior.

## Reference points
```bash
cd ~/nemo_gym/Gym
git rev-parse HEAD
python run_standalone.py --help
python aggregate.py --help
find benchmarks/owasp_llm02 resources_servers/owasp_llm02 k8s -maxdepth 3 -type f
```

## LLM06 implementation

`benchmarks/owasp_llm06/` contains all 4,220 version 3 ASB source rows, including 587 cases with reviewed scoring adapters and 3,633 explicitly unscorable cases; prior versions are archived, with separate scoring and coverage smoke inputs. `resources_servers/owasp_llm06/` provides a native Gym trace verifier and an in-memory tool loop. `run_standalone.py --risk llm06` verifies model tool support before evaluation; `aggregate.py --risk llm06` produces JSON and Markdown. Reconciliation task completion defaults to inspection coverage; an optional judge can evaluate justified outcomes without changing safety checks. [Scope and commands](../benchmarks/owasp_llm06/README.md).

## LLM10 implementation

LLM10 now has 3,000 deterministic resource-policy cases (500 per scenario), source preparation,
a typed Gym verifier, and model-driven bounded mock-tool execution. Shared runner/aggregation
support `--risk llm10`; results use `results/llm10/<model>/<run-id>/` with JSON and Markdown.
The manifest `k8s/llm10-job.yaml` generates unique run IDs and is launched manually.
[Source restrictions, measurement limits and commands](../benchmarks/owasp_llm10/README.md).
