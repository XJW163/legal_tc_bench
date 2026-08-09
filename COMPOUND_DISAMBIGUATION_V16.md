# Context-sensitive defined-term occurrence validation (v1.6)

Version 1.6 refines the `HARD / SOFT / NONE` decision made before glossary or placeholder injection.
The v1.5 validator solved `Cause/cause` and `Code` inside other statute names, but treated almost every adjacent capitalised word as ambiguous. This over-produced SOFT occurrences.

## Decision policy

### HARD: external syntax does not change the term identity

```text
Executive's Base Salary
Your Sign-On Bonus
Annual LTIP
Existing LTIP
Gilead Leadership
Tanger Team
```

Only the defined-term core span is protected. The possessive, determiner, or external modifier remains outside the placeholder.

### NONE: the span belongs to another complete named expression

```text
Chief Executive Officer       -> Executive is not the defined party term
Executive Vice President      -> Executive is a title component
Defend Trade Secrets Act      -> Trade Secrets is part of a statute name
Gilead Sciences               -> Gilead is part of the full company name
Tanger Properties             -> Tanger is part of the full entity name
Century Bank                  -> Bank is part of the entity name
Savings Plan                  -> Plan is a different named plan head
```

### SOFT: context remains genuinely ambiguous

```text
Sanctioned Person
```

The occurrence is passed only as a conditional glossary hint. It is not replaced by a placeholder and does not create a mandatory-compliance failure.

## Punctuation boundaries

Compound extraction no longer crosses punctuation. These are separate contexts:

```text
Benefits. Gilead ...
Bonus. Tanger ...
Inc. Severance Plan ...
```

A sentence-final period cannot turn the previous heading word into a modifier of the next defined term.

## Regenerate the pilot

The validator version changed, so use a new output directory or `--overwrite`:

```bash
legal-tc validate-term-occurrences \
  data/annotated \
  data/occurrences_validated_v16_pilot \
  --max-documents 5
```

Inspect the summary:

```bash
cat data/occurrences_validated_v16_pilot/occurrence_validation_manifest.json
```

Inspect all non-HARD occurrences and their new context classes:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/occurrences_validated_v16_pilot")
for path in sorted(root.glob("*.occurrence-validation.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    for term in data.get("terms", []):
        for occ in term.get("occurrences", []):
            if occ.get("control_eligibility") == "HARD":
                continue
            print("=" * 90)
            print("document:", data["document_id"])
            print("term:", term["source_term"])
            print("paragraph:", occ.get("paragraph_id"))
            print("surface:", repr(occ.get("surface")))
            print("eligibility:", occ.get("control_eligibility"))
            print("status:", occ.get("usage_status"))
            print("class:", occ.get("context_classification"))
            print("context:", occ.get("compound_context"))
            print("reasons:", occ.get("validation_reasons"))
PY
```

Expected qualitative changes from the v1.5 pilot:

- `Executive's Base Salary`, `Executive's Annual Bonus`, `Your Sign-On Bonus`, `Annual LTIP`, and `Existing LTIP` move from SOFT to HARD;
- `Chief Executive Officer`, `Executive Vice President`, `Defend Trade Secrets Act`, `Gilead Sciences`, `Tanger Properties`, `Century Bank`, and named `... Plan` expressions move from SOFT to NONE;
- `Benefits. Gilead` and similar cross-sentence false compounds become standalone HARD occurrences;
- `Sanctioned Person` remains SOFT;
- lowercase/common-word occurrences and `Code` inside different legal names remain NONE.

## Rerun glossary control

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-glossary-v16-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode glossary \
  --occurrence-validation data/occurrences_validated_v16_pilot \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

## Run placeholder control after inspection

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v16-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --occurrence-validation data/occurrences_validated_v16_pilot \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

The validator remains deliberately high precision. A case-mismatched occurrence is still excluded rather than promoted automatically. A later semantic classifier can recover those cases without weakening placeholder safety.

The manifest also reports aggregate counts for usage status, validation reason, and context classification across all processed documents. These counts make it possible to compare v1.5 and v1.6 without scanning every occurrence artifact.
