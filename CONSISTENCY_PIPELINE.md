# Legal terminology consistency pipeline (v0.7)

This release extends the existing project **after occurrence-level rendering extraction**.
No retranslation and no re-extraction are required.

## New CLI commands

### 1. `aggregate-renderings`
Groups occurrence rows by `(document_id, term_id)` and computes lexical/formal
consistency statistics.

```bash
legal-tc aggregate-renderings data/renderings data/aggregated
```

For a multi-model directory, one file is produced per translation experiment:

```text
data/aggregated/<experiment>.aggregated.json
```

### 2. `llm-judge-consistency`
Only terms with **more than one distinct explicit rendering** are sent to the LLM.
Single-rendering groups are automatically marked exact-consistent.

```bash
legal-tc llm-judge-consistency \
  data/aggregated data/judgments \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 5 \
  --confidence-threshold 0.8 \
  --max-retries 5 \
  --timeout 300
```

The command uses atomic JSON checkpoints. Rerun the same command after a network,
SSH, Python, or vLLM interruption. Existing completed batches are restored.

Semantic labels:

- `EXACT_CONSISTENT`
- `ACCEPTABLE_VARIANT`
- `LEGAL_INCONSISTENCY`
- `UNRESOLVED`

The judge prompt explicitly separates **translation consistency** from general
translation accuracy. A consistently wrong translation is not automatically counted as
an inconsistency.

### 3. `final-metrics`
Combines formal variation and semantic/legal judgments.

```bash
legal-tc final-metrics data/aggregated data/judgments data/final_metrics
```

Reported metrics include:

- DTR macro and micro
- TVR macro
- entropy and normalized entropy
- Semantic Legal Consistency Rate (SLCR)
- Legal Inconsistency Rate (LIR)
- omission rate
- explicit-rendering coverage
- unresolved rate
- critical-term inconsistency rate
- critical-term risk rate
- Weighted Legal Consistency Score (WLCS; summary metric only)

### 4. `compare-models`

```bash
legal-tc compare-models \
  data/final_metrics \
  data/model_comparison.json \
  --csv data/model_comparison.csv
```

This produces a paper-ready cross-model table.

## Recommended execution from the user's current state

Because occurrence counts and Chinese rendering extraction are already complete, run
only these four commands:

```bash
legal-tc aggregate-renderings data/renderings data/aggregated

legal-tc llm-judge-consistency \
  data/aggregated data/judgments \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 5 --confidence-threshold 0.8 --max-retries 5 --timeout 300

legal-tc final-metrics data/aggregated data/judgments data/final_metrics

legal-tc compare-models \
  data/final_metrics data/model_comparison.json \
  --csv data/model_comparison.csv
```

## v0.8 judge robustness

`llm-judge-consistency` now retries not only network/JSON failures but also strict
validation failures such as missing `variant_id`s. Every request explicitly lists the
required non-dominant variant IDs. If a model returns the dominant variant redundantly,
it is ignored safely; missing non-dominant variants still trigger a retry.

For smaller judges such as `gpt-oss-20b`, `--batch-size 1` or `2` is recommended if
schema omissions remain frequent. Checkpoint files remain compatible with v0.7, so a
rerun continues from already completed batches.
