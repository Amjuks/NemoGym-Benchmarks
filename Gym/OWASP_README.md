# NeMo Gym OWASP Evaluation — Developer Handbook

**Status:** This handbook describes the implementation discussed and exercised on the SoketLab cluster; it is **not** a live audit of your remote checkout. Verify paths, signatures and command flags against the checkout before changing code.

## Start here
1. [Current implementation](docs/01-current-system.md)
2. [Developer quickstart](docs/02-quickstart.md)
3. [Datasets and conversion](docs/03-datasets.md)
4. [Adding an OWASP risk](docs/04-new-risk.md)
5. [Verifiers and tests](docs/05-verifiers.md)
6. [Kubernetes execution](docs/06-kubernetes.md)
7. [Results and reporting](docs/07-results-reporting.md)
8. [Troubleshooting](docs/08-troubleshooting.md)
9. [Standards and review checklist](docs/09-standards.md)

The companion `.gitignore` is intended for the **parent `nemo_gym/` folder**. See `docs/10-git-layout.md` before initializing a parent repository.

## Architecture

```text
source datasets -> benchmarks/owasp_llmXX/prepare.py
                -> prepared Gym JSONL
                -> run_standalone.py (current LLM02 execution path)
                -> target model via LiteLLM
                -> resources_servers/owasp_llmXX/app.py verifier
                -> per-shard JSONL -> aggregate.py -> standardized results
```

**Important:** `run_standalone.py` and `aggregate.py` are project-specific convenience scripts, **not NVIDIA's standard Gym evaluation CLI**. The preferred longer-term direction is native Gym evaluation and verification. Do not confuse the two interfaces.

## Repository boundaries
- `Gym/`: upstream NVIDIA NeMo Gym checkout plus project-specific modifications.
- `Gym/benchmarks/owasp_llm02/`: current prepared dataset and preparation code.
- `Gym/resources_servers/owasp_llm02/`: LLM02 verifier.
- `Gym/k8s/`: Kubernetes Job definitions.
- `Gym/results/`: recommended centralized result location; typically ignored by Git.

## Security and reproducibility
Use synthetic/sanitized secrets, never commit credentials, and redact report examples. Record the exact Git commit, dataset versions, verifier version, model ID, inference endpoint identifier, run date, and parameters for every published evaluation.
