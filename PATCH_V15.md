# v1.5.0 patch summary

- Added `validate-term-occurrences`.
- Added deterministic definition-use disambiguation with `HARD`, `SOFT`, and `NONE` control levels.
- Excluded lowercase/common-word uses such as `cause` from the defined term `Cause`.
- Excluded `Code` when it is part of another legal name such as `United States Code` or `Internal Revenue Code`.
- Added explicit definiendum recognition.
- Added conditional glossary hints for ambiguous capitalised compounds.
- Integrated precomputed or inline occurrence validation into `controlled-translate`.
- Added stale-validation source-signature checks.
- Placeholder mode now applies only to `HARD` occurrences.
- Paragraphs with no HARD occurrence are no longer counted as placeholder successes.
- Added exact/approved/missing glossary-compliance details.
- Filtered informal allowed variants from formal-contract compliance.
- Added HARD/SOFT/NONE and strategy statistics to manifests.
- Added tests for the observed `Cause/cause` and `Code` failure modes.
