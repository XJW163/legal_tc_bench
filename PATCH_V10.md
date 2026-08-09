# v1.0.0 changes

- Added deterministic rendering-boundary validation.
- Added longest-term containment exclusion.
- Added detection of untranslated Latin terms, over-wide Chinese spans, one-character ambiguous spans, mixed-Latin spans, and source-side outside modifiers.
- Added resumable LLM minimal-span repair for suspicious occurrences only.
- Added term definition and representative contexts to legal-consistency judge payloads.
- Made non-empty group and variant rationales mandatory for newly generated judgments.
- Aggregation now separates raw, eligible, and validation-excluded occurrences.
- Added incremental reuse of old judgments for unchanged groups.
- Added validation exclusion rate to final model comparison.
- Restored translation experiment-signature safety check.
- Removed an accidental live vLLM call from `tests/test_renderings.py`.
