# Validation performed

- All Python files passed `python -m compileall`.
- All shell files passed `bash -n`.
- Standalone scripts were executed against a synthetic two-document fixture covering:
  - evaluation audit and failure export;
  - deterministic terminology metrics;
  - direct A/B subset construction and analysis;
  - blinded human-review generation and analysis;
  - low-cost command generation;
  - fixed-subset extraction;
  - ablation document selection and command generation;
  - qualitative case export;
  - efficiency/cost aggregation;
  - paper-table generation.
- No external API was called during validation.
