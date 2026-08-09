# Placeholder control v1.4

## Recommended pilot command

Use a new experiment name because the placeholder strategy and control signature changed.

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v14-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

## Output interpretation

- `placeholder_exact`: initial output preserved every unique occurrence token.
- `placeholder_repaired`: a minimum-edit repair restored the token inventory.
- `glossary_fallback`: hard conservation failed; the same canonical glossary was used as a soft fallback.

`constraint_compliance=true` has different strength by strategy:

- under `placeholder_exact` or `placeholder_repaired`, it means every protected occurrence token was conserved and deterministically restored;
- under `glossary_fallback`, it only means every required canonical rendering appeared at least once.

## Quick audit

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path("data/translations_controlled/gpt-oss-20b-placeholder-v14-pilot")
strategies = Counter()
missing = Counter()
for path in root.glob("*.controlled.translation.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("translations", []):
        strategies[row.get("constraint_strategy_used", "legacy")] += 1
        diag = row.get("placeholder_diagnostics") or {}
        for token in diag.get("missing", []):
            missing[token.split("_")[0]] += 1
print("strategies:", strategies)
print("most frequently missing term bases:", missing.most_common(20))
PY
```
