# 04 — Adding a New OWASP Risk

Each risk should be independently runnable. Do **not** extend the LLM02 verifier with unrelated logic.

## Intended layout
```text
Gym/
├── benchmarks/
│   ├── owasp_llm02/          # implemented
│   ├── owasp_llm06/          # standalone mock-tool implementation
│   └── owasp_llm10/          # add when implemented
├── resources_servers/
│   ├── owasp_llm02/
│   ├── owasp_llm06/
│   └── owasp_llm10/
├── k8s/
└── results/
```
Future directories shown here are **planned**, not confirmed to exist.

## New-risk implementation procedure
1. Create an issue/spec: OWASP ID, threat definition, model capabilities, allowed tools, measurable failure conditions and exclusions.
2. Review the **pinned** NeMo Gym scaffold and examples. Check `gym --help` and installed docs; do not assume flags from a newer release.
3. Add `benchmarks/owasp_llmXX/` with `prepare.py`, dataset manifest/README, prepared JSONL and native config where required.
4. Add `resources_servers/owasp_llmXX/` with typed verifier request/response, `async verify(...)` and tests.
5. Keep scoring deterministic when possible; add a separately configured judge only for semantic decisions.
6. Add model inference/execution configuration; for agentic risks use **sandboxed mock tools**, not live side-effecting integrations.
7. Add an isolated Kubernetes Job manifest or parameterized template; output to a unique `risk/model/run-id`.
8. Add a reporting adapter mapping risk-specific verifier outcomes to standard **execution health** fields and separately named risk metrics.
9. Run unit tests, a 10-case smoke test, a small balanced run, then full evaluation.
10. Document model, data, verifier, tool simulator and benchmark versions.

## LLM06: Excessive Agency
- Use mock email, file, payments, admin and other scoped tools only.
- Define authorized permissions and expected-action traces per case.
- Score tool calls against scope, confirmation/approval requirements and task completion.
- Include safe controls: legitimate actions should succeed.
- Do not treat an agent's *claim* to have acted as a real tool action.

## LLM10: Unbounded Consumption
- Establish explicit per-case token, wall-time, request, tool-call and recursion budgets.
- Capture resource usage and stop conditions.
- Separate attack-induced runaway behavior from provider throttling, 503s and infrastructure limits.
- Set hard caps in the harness to prevent unbounded spend.

## Definition of done
Prepared data reproducible; verifier tests pass; mock tools isolated; no ground-truth leakage; end-to-end run scores successfully; aggregate has explicit denominators; documentation and review complete.

The LLM06 implementation and its current exclusions are documented in [the benchmark README](../benchmarks/owasp_llm06/README.md). Do not treat its reconciliation workflow proxy as semantic task success or generalize its permission checks to unsupported source environments.

LLM10 is implemented using the same runner and aggregator: see
[`benchmarks/owasp_llm10/README.md`](../benchmarks/owasp_llm10/README.md). Its bounded harness
must supply trusted telemetry to the deterministic verifier; a text-only response cannot
measure request loops, tool quotas or context admission.
