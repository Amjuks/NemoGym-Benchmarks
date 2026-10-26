# LLM06 resources server

`app.py` provides `OwaspLlm06VerifyRequest`, `OwaspLlm06Verifier`, and `OwaspLlm06ResourcesServer` using the native Gym base classes. Verification is stateless: raw tool calls are replayed from each case's initial mock state, bound to the model response's function calls, and graded deterministically.

`mock_tools.py` implements declarative sandbox tools, permissions, source-backed prerequisites, trusted approval state, and completion checks. `agent.py` supplies the Responses API capability check and tool loop used by the shared OWASP standalone runner. No dataset source is executed and there are no host or network tool effects.

See the [benchmark documentation](../../benchmarks/owasp_llm06/README.md) for data scope, unavailable metrics, tests, smoke commands, and cluster execution.

Version 3 includes every source row. A case without reviewed scoring adapters returns `reward=null`, `evaluation_completed=false`, and unknown safety/completion metrics. Supported partial tools may execute, but unavailable source implementations return explicit errors without effects. These rollouts remain separate from scored cases and API failures in the standalone reports.

## Independent standalone judge

`standalone.py` loads standard Gym-style YAML and constructs a separate `NeMoGymAsyncOpenAI` client for `judge_base_url`, `judge_api_key`, and `judge_model_name` (default `gpt`). It has no policy-setting fallback and starts no servers. Run the driver with `--judge-config resources_servers/owasp_llm06/configs/owasp_llm06.yaml --judge-env-file env.yaml`.

`CompletionJudgeVerifier` in `app.py` handles judging inside `verify()`, after deterministic replay. `completion.py` constructs evidence, parses strict JSON, and prevents a judge verdict from overriding safety decisions. Standalone calls use Gym's `reraise_judge_errors`; native deployments can still use `OwaspLlm06ResourcesServer` and `call_judge` through a named model server. Both share the same verification logic.

`benchmarks/response_utils.py` normalizes the gateway's token-usage envelope before Gym Responses validation. Transport failures are unscored and retain the policy response/trace; invalid received verdicts have separate parsing diagnostics. Exact outcome/approval checks and cases without reviewed adapters skip the judge. See the [official guide's external endpoint alternative](https://docs.nvidia.com/nemo/gym/v0.7.0/build-verifiers/verification-patterns/llm-as-judge/) and benchmark README for complete settings and commands.
