# Definition-grounded terminology consistency control (v1.3)

This release changes the project from evaluation-only toward an active consistency-control method.

## New commands

### 1. Build document-level legal term memory

```bash
legal-tc build-term-memory \
  data/annotated \
  data/term_memory \
  --aggregated data/aggregated_clean \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 4 \
  --confidence-threshold 0.8 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

Each memory entry contains:

- canonical Chinese rendering;
- legal-scope summary grounded in the contract definition;
- approved semantic variants and usage conditions;
- risky variants such as scope narrowing, broadening or category shift;
- confidence and rationale.

`--aggregated` is optional. When supplied, observed renderings from previous baseline translations are passed as evidence, not as ground truth.

### 2. Controlled translation with hard placeholders

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory \
  data/translations_controlled \
  --experiment qwen3-32b-placeholder-v1 \
  --model Qwen3-32B \
  --base-url http://10.12.143.51:8000/v1 \
  --constraint-mode placeholder \
  --context-paragraphs 2 \
  --workers 2 \
  --max-documents 5 \
  --max-retries 5 \
  --timeout 600
```

A source occurrence such as `Person` is replaced by a stable token such as `【术语0002】`. The model must preserve all tokens with exactly the same counts. The application then deterministically restores the approved Chinese term.

This yields a hard lexical-consistency guarantee for successfully validated paragraphs.

### 3. Glossary-prompt ablation

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory \
  data/translations_controlled \
  --experiment qwen3-32b-glossary-v1 \
  --model Qwen3-32B \
  --base-url http://10.12.143.51:8000/v1 \
  --constraint-mode glossary
```

This mode does not replace source terms. It only injects mandatory source-to-target mappings and records paragraphs where the required canonical terms are absent. It is intended as a baseline/ablation, not the final method.

## Recommended immediate experiment

Run a five-document pilot with the same translation model:

1. existing zero-English baseline;
2. glossary control;
3. placeholder control.

Then feed all three outputs through the existing occurrence-review, rendering extraction, cleanup, aggregation, judgment and final-metrics pipeline. Compare consistency improvement, placeholder retry rate, translation errors, latency and token cost before scaling to all 68 documents.

## v1.5 prerequisite: validate defined-term use

Do not inject placeholders from raw case-insensitive string matches. First run:

```bash
legal-tc validate-term-occurrences \
  data/annotated \
  data/occurrences_validated
```

Then pass the artifact to both controlled-translation ablations:

```bash
--occurrence-validation data/occurrences_validated
```

`HARD` occurrences receive mandatory controls, `SOFT` occurrences receive conditional hints, and `NONE` occurrences are excluded. See `DEFINITION_USE_DISAMBIGUATION_V15.md`.
