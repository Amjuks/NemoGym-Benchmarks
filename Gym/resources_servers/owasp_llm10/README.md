# OWASP LLM10 resource verifier

See [preparation, policy, execution and reporting documentation](../../benchmarks/owasp_llm10/README.md).

`app.py` exposes the installed Gym typed async verifier. `budget.py` owns validated per-case
quotas and measurements; `agent.py` executes a model-driven bounded tool loop; `mock_tools.py`
reuses LLM06 tool schemas and in-memory reads. `standalone.py` preserves raw telemetry while
adapting response envelopes for the shared runner. No resource check uses an LLM judge.

The Gym resource-server config is `configs/owasp_llm10.yaml`. Caller-supplied telemetry must
come from the trusted harness, never from model output. Generic rollouts without resource
telemetry return null reward and `mask_sample=True`.
