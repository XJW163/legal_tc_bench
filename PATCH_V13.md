# v1.3.0 — From benchmark-only evaluation to active consistency control

## Added

- `build-term-memory`: definition-grounded document-level terminology planning.
- `controlled-translate --constraint-mode glossary`: prompt-glossary ablation.
- `controlled-translate --constraint-mode placeholder`: hard lexical constraints using Chinese-safe placeholders.
- Paragraph-level checkpoint/resume for controlled translation.
- Exact placeholder-count validation and deterministic canonical-term restoration.
- Per-paragraph audit metadata, retry counts, compliance status and term occurrence mapping.
- Optional use of `aggregated_clean` as candidate evidence for terminology planning.

## Research role

The previous pipeline remains the evaluator. v1.3 adds the first actual intervention:

1. infer canonical translations from contract definitions;
2. inject only occurrence-relevant controls;
3. force stable output via placeholders;
4. evaluate the controlled output using the existing rendering and legal-consistency pipeline.

The online semantic verifier and minimal-repair controller are intentionally left for the next release, after the planner and hard-control pilot are validated.
