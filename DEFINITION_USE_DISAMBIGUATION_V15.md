# Definition-use disambiguation and risk-adaptive term control (v1.5)

Version 1.5 prevents a defined-term string match from being treated automatically as a legal-term use.
This is required before placeholder injection. For example:

```text
Cause              defined term: termination for cause
cause injury       ordinary verb: do not control

Code               defined shorthand for the Internal Revenue Code
United States Code a different legal name: do not replace its final word
```

## 1. Validate candidate occurrences

Run the deterministic validator before controlled translation:

```bash
legal-tc validate-term-occurrences \
  data/annotated \
  data/occurrences_validated_pilot \
  --max-documents 5
```

The output contains one `*.occurrence-validation.json` file per document and a manifest.
Every candidate occurrence receives:

```json
{
  "usage_status": "DEFINED_TERM_USAGE",
  "control_eligibility": "HARD",
  "validation_confidence": 0.98,
  "definition_role": "OTHER",
  "validation_reasons": ["EXACT_CASE_STANDALONE_MATCH"]
}
```

Control levels:

- `HARD`: exact, case-sensitive, standalone defined-term use; safe for a unique placeholder.
- `SOFT`: possible defined-term use inside a larger capitalised expression; use only a conditional glossary hint.
- `NONE`: lowercase/common-word use, another legal-name component, invalid span, or source mismatch; no control.

The current deterministic rules include:

1. case-sensitive defined-term matching (`Cause` is distinct from `cause`);
2. explicit definiendum recognition (`“Code” shall mean ...`);
3. exclusion of `Code` inside names such as `Internal Revenue Code`, `United States Code`, `U.S. Code`, `Bankruptcy Code`, and `Commercial Code`;
4. conservative `SOFT` handling for capitalised modifiers such as `Sanctioned Person`;
5. source-span and source-slice integrity checks;
6. source signatures that reject stale validation files.

## 2. Inspect the pilot

```bash
cat data/occurrences_validated_pilot/occurrence_validation_manifest.json
```

Print all non-HARD occurrences:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/occurrences_validated_pilot")
for path in sorted(root.glob("*.occurrence-validation.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    for term in data.get("terms", []):
        for occ in term.get("occurrences", []):
            if occ.get("control_eligibility") == "HARD":
                continue
            print("=" * 90)
            print(data["document_id"], term["source_term"])
            print("paragraph:", occ.get("paragraph_id"))
            print("surface:", repr(occ.get("surface")))
            print("eligibility:", occ.get("control_eligibility"))
            print("status:", occ.get("usage_status"))
            print("context:", occ.get("compound_context"))
            print("reasons:", occ.get("validation_reasons"))
PY
```

For the reported pilot, lowercase `cause` should be `NONE`, while `Code` inside `Internal Revenue Code` and `United States Code` should also be `NONE`.

## 3. Run glossary control with validated occurrences

Use a new experiment name because the control signature has changed:

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-glossary-v15-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode glossary \
  --occurrence-validation data/occurrences_validated_pilot \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

Only `HARD` occurrences are mandatory and count toward glossary compliance. `SOFT` occurrences are explicitly described to the model as conditional hints and never create a compliance failure. `NONE` occurrences are omitted from the prompt.

Glossary checking now records:

- `EXACT_CANONICAL`;
- `APPROVED_VARIANT` for formal variants in term memory;
- `MISSING_REQUIRED_RENDERING`.

Variants whose usage condition says informal, oral, or internal communication are not accepted for formal-contract compliance.

## 4. Run placeholder control

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v15-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --occurrence-validation data/occurrences_validated_pilot \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

Behavior by occurrence level:

```text
HARD -> occurrence-unique placeholder
SOFT -> conditional glossary hint; source remains unchanged
NONE -> no control
```

Paragraph strategy values are now auditable:

- `placeholder_exact`: paragraph had at least one HARD occurrence and all placeholders survived initially;
- `placeholder_repaired`: HARD placeholder inventory was repaired;
- `glossary_fallback`: hard placeholder control failed and mandatory glossary fallback was used;
- `conditional_glossary`: paragraph had only SOFT occurrences;
- `uncontrolled`: paragraph had no eligible term occurrence.

A paragraph without a HARD occurrence is no longer counted as a placeholder success.

## 5. New statistics

The document and experiment manifests include:

```text
hard_controlled_term_occurrences
soft_controlled_term_occurrences
excluded_term_occurrences
placeholder_exact_paragraphs
placeholder_repaired_paragraphs
conditional_glossary_paragraphs
uncontrolled_paragraphs
placeholder_fallback_paragraphs
glossary_noncompliance_paragraphs
```

Each translated paragraph also stores:

- the validated status and reasons of controlled terms;
- excluded candidate occurrences and why they were excluded;
- detailed glossary compliance results;
- whether an approved formal variant, rather than the exact canonical form, was used.

## 6. Inline mode

`--occurrence-validation` is optional. If it is omitted, `controlled-translate` runs the same deterministic validator in memory and records:

```text
occurrence_validation_source = inline-deterministic
```

Precomputation is recommended for experiments because it creates an auditable artifact and makes manual inspection possible.

## 7. Recommended next action

First regenerate the five-document glossary pilot with v1.5 and compare the old nine apparent failures. The expected changes are:

- six lowercase `cause` false positives disappear;
- `Code` inside `United States Code` disappears;
- `Code` inside `Internal Revenue Code` is not hard-controlled;
- the genuine untranslated/incorrect `Code` definiendum remains visible;
- `Bank` short-form behavior remains visible unless explicitly approved in term memory.

Only after this check should the placeholder v1.5 pilot be run.
