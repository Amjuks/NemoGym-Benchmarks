# NeMo Gym evaluation quickstart

Clone the repository and start in its `nemo_gym/` folder. Replace `your-model`, `http://SERVER:PORT/v1`, and the output folder with your settings.

## Setup

Requires Python 3.13.14+ and `uv`. Install dependencies and activate the environment:

```bash
cd Gym
uv sync --extra dev
uv pip install --directory resources_servers/iheval --python ../../.venv/bin/python -r requirements.txt
uv pip install --directory resources_servers/toolsandbox --python ../../.venv/bin/python -r requirements.txt
source .venv/bin/activate
python -m nltk.downloader punkt punkt_tab
```

Run the remaining commands from `Gym/` in the same terminal.

## Configure

Copy the example configuration once, then edit `env.yaml` with your endpoint, API key,
model, simulator, judge, and token budgets. If it already exists, add the example's
`evaluation` section to it. Gym loads this private file automatically.

```bash
cp -n env.example.yaml env.yaml
```

Choose a fresh output folder. Reuse it when resuming the same run.

```bash
RUN='results/my-model-run'
```

Prepare the run folder, configuration files and reporting manifest. Native dataset builders
download the public source data when it is not already cached.

```bash
python scripts/prepare_evaluation_run.py --output "$RUN"
```

Use an OpenAI-compatible endpoint with tool calling for ToolSandbox. The model, simulator, and judge must be available at that endpoint.

Change settings in `env.yaml` before preparing a new run. Optional command-line flags,
such as `--model`, override those settings. Generated configs reference the API key;
they do not copy it into the results folder.

## Run

Run IHEval. Repeat this command to resume an interrupted run.

```bash
gym eval run --config "$RUN/iheval.yaml" --output "$RUN/llm02_iheval/rollouts.jsonl" --resume
```

Run ToolSandbox. Repeat this command to resume an interrupted run.

```bash
gym eval run --config "$RUN/toolsandbox.yaml" --output "$RUN/llm06_toolsandbox/rollouts.jsonl" --resume
```

Run safety judging after IHEval finishes.

```bash
gym eval reverify --config "$RUN/aegis.yaml" \
  --inputs "$RUN/llm02_iheval/rollouts_materialized_inputs.jsonl" \
  --rollouts "$RUN/llm02_iheval/rollouts.jsonl" \
  --output "$RUN/llm02_iheval/aegis/rollouts.jsonl" --concurrency 16 --resume
```

## Export results

Generate the overall benchmark-summary CSVs in `$RUN/csv/`.

```bash
python scripts/export_evaluation_results.py "$RUN" --summary-only
```

Generate the full-benchmark risk CSVs and column explanations in `$RUN/risk_csv/`.

```bash
python scripts/export_risk_results.py "$RUN"
```

## Kubernetes

For cluster runs, adapt [Gym/k8s/job-native-full.yaml](Gym/k8s/job-native-full.yaml) with the run's
config/output paths, worker node, host paths, and Python runtime paths. Prepare the run on the worker checkout at the same paths mounted inside the jobs.

Create the evaluation namespace.

```bash
kubectl apply -f k8s/namespace-owasp.yaml
```

Place the private `env.yaml` in the worker's mounted `Gym/` checkout as well.

Start both evaluations.

```bash
kubectl apply -f k8s/job-native-full.yaml
```

Check progress.

```bash
kubectl get jobs -n owasp-bench
```

Use a fresh run folder and distinct Job names for each model. Run safety judging and CSV export with the commands above once the jobs complete.

Generated datasets, local credentials and evaluation results are excluded from Git and Docker build contexts.
Upstream documentation and third-party notices remain in `Gym/`; see [LICENSE](LICENSE) and [Gym/ATTRIBUTIONS.md](Gym/ATTRIBUTIONS.md).
