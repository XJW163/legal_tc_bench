# Patch v1.1.0

`llm-repair-renderings` no longer aborts the full corpus when one occurrence
repeatedly returns malformed, empty, or non-exact model output.

Changes:

- Failed multi-item batches are isolated to per-occurrence calls.
- A single unusable model result is stored as `REPAIR_FAILED_RETAINED`.
- The original rendering is preserved for audit but excluded from consistency metrics.
- `repair.audit.json` reports `repair_failed_retained_count`.
- `--retry-failed` retries only retained failures, without rerunning successful rows.
- Conservative exact-span recovery tolerates whitespace/NFKC differences only; it
  never substitutes a semantically similar phrase that is absent from the target.
- Real transport/API failures still stop the job after retries, allowing a safe resume.
