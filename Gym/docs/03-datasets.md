# 03 — Adding and Maintaining Datasets

## Dataset inventory
| Source | Current converted records | Notes |
|---|---:|---|
| PasswordEval | 1,000 | Password/contextual-integrity scenarios |
| PEEP | 1,984 | Privacy/contextual-integrity scenarios |
| PrivAwareBench | 248 | Privacy awareness scenarios |
| **Total** | **3,232** | Current LLM02 prepared set |

Other candidate datasets discussed previously include ConfAIde, DecodingTrust Privacy, PrivacyLens, Comparison Dataset and ALERT. **They are not confirmed as integrated**; inspect their schemas, licenses and relevance before adding.

## Steps for a new dataset
1. Choose a risk and its existing verifier **first**. Read its typed request schema and expected task fields.
2. Record the original source, dataset revision/commit, license, split, row count and source citation.
3. Map the source fields to the verifier's required fields; preserve source labels and provenance.
4. Construct model-visible messages under `responses_create_params.input`; keep ground truth out of model-visible content unless the scenario explicitly requires it.
5. Assign stable case IDs; deduplicate without erasing meaningful variations.
6. Validate JSONL parsing and verifier-required fields; sample cases by hand.
7. Test positive, negative, authorized and unauthorized cases. Check a known reference result if available.
8. Document included/excluded counts and reasons.
9. Rebuild the prepared dataset reproducibly; bump dataset version/hash and rerun the smoke test.

## Shape illustration — NOT the full LLM02 schema
```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Example prompt for the model"}
    ]
  }
}
```
The verifier-only fields must match **the actual `OwaspLlm02VerifyRequest` and preparation script**. Do not paste arbitrary `verifier_metadata` fields and assume the verifier reads them.

## Recommended location
```text
benchmarks/owasp_llm02/
├── README.md              # dataset provenance, counts, reward semantics
├── config.yaml            # Gym configuration, if used by pinned revision
├── prepare.py             # deterministic source-to-JSONL conversion
└── data/
    ├── example.jsonl      # prepared records
    └── balanced_90.jsonl  # smoke/stratified sample
```

## Quick checks
```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path('benchmarks/owasp_llm02/data/example.jsonl')
rows = [json.loads(line) for line in p.open() if line.strip()]
print('rows:', len(rows))
print('dataset:', Counter(r.get('source_dataset', 'unknown') for r in rows))
print('with model input:', sum('responses_create_params' in r for r in rows))
PY
```
Also check unique case IDs and inspect any records that fail typed verifier validation.

## Reproducibility checklist
- [ ] Exact source revision and license recorded
- [ ] Stable row IDs and split names
- [ ] No live credentials or unnecessary personal data
- [ ] Model input and verifier ground truth separated
- [ ] Positive and negative controls included
- [ ] Conversion logs accepted/rejected counts
- [ ] Verifier tests and smoke run pass
- [ ] Dataset hash/version included in run metadata

See the bundled reference documents for the Gym JSONL approach; **the precise row schema is benchmark-specific**.

## LLM06 source adapters

ASB revision `b7761ca4c4e7a5937f0b6ad11dd24ce19163c454` contributes all **4,220 source rows** in version 3: 3,464 training and 756 augmented. Every row retains its full source payload, identity, and hash; duplicate rows remain separate cases. There are zero exclusions. Of these, 587 cases have reviewed scoring adapters; 3,633 retain null rewards and explicit reasons in `data/unscored.jsonl`. Unsupported source tools return errors without effects; missing interfaces are not invented. Versions 1 and 2 are archived under `data/v1/` and `data/v2/`. Pinned checksums, schemas, provenance, and counts appear in the [preparation report](../benchmarks/owasp_llm06/data/preparation_report.json). Source Python is parsed only, never executed. Run `.venv/bin/python -m benchmarks.owasp_llm06.prepare` to regenerate.

## LLM10 task seeds

`python -m benchmarks.owasp_llm10.prepare` downloads pinned Hugging Face Alpaca and Instruct-v3
revisions and produces 3,000 distinct cases plus a balanced 20-case smoke file. Each category
has 500 cases; 1,500 use Alpaca and 1,500 use Instruct-v3 (750 train, 750 published test).
Instruct-v3 has no validation split: its test partition is explicitly used as validation.
These are passages for deterministic derived tasks and synthetic lookups, not vulnerability labels.
Resource-policy outcomes supply ground truth. CC BY-NC 4.0 / CC BY-SA 3.0 source-derived artifacts
stay local and Git-ignored; code remains Apache-2.0. See the [LLM10 preparation report and provenance
instructions](../benchmarks/owasp_llm10/README.md).
