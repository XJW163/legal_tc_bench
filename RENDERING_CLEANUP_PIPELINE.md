# Rendering cleanup and incremental re-judgment

This version adds a cleanup stage between occurrence-level rendering extraction and
aggregation. It addresses over-wide target spans such as `any Person -> 任何实体`,
ambiguous one-character alignments such as `Person -> 方`, untranslated terms such as
`Software -> Software`, and source occurrences nested in a longer benchmark term.

## 1. Deterministic validation

```bash
legal-tc validate-renderings \
  data/renderings \
  data/annotated \
  data/renderings_validated
```

The command does not call an LLM. It enriches every row with:

- `source_paragraph`
- `source_marked_context`, where the exact source occurrence is wrapped in `<TERM>`
- `definition_context`
- `validation_status`
- `validation_reasons`
- `include_in_consistency`
- `needs_rendering_repair`

Important validation statuses:

- `VALID`
- `NESTED_IN_LONGER_TERM` — excluded from consistency metrics
- `UNTRANSLATED`
- `POSSIBLY_OVER_WIDE`
- `AMBIGUOUS_ALIGNMENT`
- `MIXED_LATIN`
- `MISSING_RENDERING`

Inspect the manifest before calling the LLM:

```bash
cat data/renderings_validated/validation_manifest.json
```

## 2. Targeted minimal-span repair

Only suspicious rows are sent to the judge model:

```bash
legal-tc llm-repair-renderings \
  data/renderings_validated \
  data/renderings_clean \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 2 \
  --max-retries 5 \
  --timeout 300
```

The prompt explicitly requires the smallest exact Chinese substring. For example:

- `any <TERM>Person</TERM> -> 实体`, not `任何实体`
- `such <TERM>Person</TERM> -> 主体`, not `该主体`
- `<TERM>U.S. Person</TERM> -> 美国人士`

The command supports checkpoint recovery. Re-run the same command after network,
SSH, or vLLM interruption. Do not use `--overwrite` when resuming.

Outputs retain the original candidate in `repair_previous_rendering` and add repair
confidence, rationale, and model fields.

## 3. Re-aggregate cleaned renderings

Do not overwrite the old aggregation until the cleaned result has been checked:

```bash
legal-tc aggregate-renderings \
  data/renderings_clean \
  data/aggregated_clean
```

Aggregation now:

- ignores rows with `include_in_consistency=false`;
- reports raw and eligible occurrence counts separately;
- reports `validation_excluded_occurrences`;
- carries `definition_context` into the legal-semantic judge input.

## 4. Reuse unaffected previous judgments

Most groups usually remain unchanged. Seed a new judgment directory from the old
results:

```bash
legal-tc reuse-judgments \
  data/aggregated \
  data/judgments \
  data/aggregated_clean \
  data/judgments_clean
```

A judgment is reused only when the source term, rendering strings, rendering counts,
status counts, and eligible occurrence count are unchanged. Variant IDs are remapped
by rendering. Changed groups are deliberately omitted.

## 5. Judge only changed groups

Run the normal command against the seeded directory:

```bash
legal-tc llm-judge-consistency \
  data/aggregated_clean \
  data/judgments_clean \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --batch-size 2 \
  --confidence-threshold 0.8 \
  --max-retries 5 \
  --timeout 300
```

Existing reused groups are skipped. The judge now receives the term definition and
representative source/target contexts, and every non-dominant judgment must contain a
non-empty rationale.

## 6. Final metrics

```bash
legal-tc final-metrics \
  data/aggregated_clean \
  data/judgments_clean \
  data/final_metrics_clean

legal-tc compare-models \
  data/final_metrics_clean \
  data/model_comparison_clean.json \
  --csv data/model_comparison_clean.csv
```

The final comparison includes `Validation_exclusion_rate` so alignment artifacts are
visible rather than silently discarded.
