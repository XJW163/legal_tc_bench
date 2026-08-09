#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_CONFIG="${1:-analysis_config.json}"
BASE_CONFIG="${2:-config/final_experiment.json}"
MODEL_KEY="${3:-gpt-oss-20b}"

python - "$ANALYSIS_CONFIG" "$BASE_CONFIG" <<'PY'
import json, sys
from pathlib import Path
analysis_path = Path(sys.argv[1]).resolve()
base_path = Path(sys.argv[2]).resolve()
a = json.loads(analysis_path.read_text(encoding='utf-8'))
b = json.loads(base_path.read_text(encoding='utf-8'))
root = Path(a.get('project_root', '.'))
if not root.is_absolute(): root = (analysis_path.parent / root).resolve()
def r(value):
    p = Path(value)
    return str(p if p.is_absolute() else (root / p).resolve())
paths = a['paths']
out = Path(r(paths['output_root'])) / 'ablations'
b['project_root'] = str(root)
b['paths']['annotated'] = r(paths['annotated'])
b['paths']['term_memory'] = r(paths['term_memory'])
b['paths']['occurrence_validation'] = r(paths['occurrence_validation'])
b['paths']['controlled_root'] = str(out / 'translations_controlled')
b['paths']['results_root'] = str(out / 'results')
out.mkdir(parents=True, exist_ok=True)
(out / 'ablation_config.json').write_text(json.dumps(b, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(out)
PY

OUT_ROOT=$(python - "$ANALYSIS_CONFIG" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); d=json.loads(p.read_text())
r=Path(d.get('project_root','.')); r=(p.parent/r).resolve() if not r.is_absolute() else r
o=Path(d['paths']['output_root']); print((r/o/'ablations').resolve() if not o.is_absolute() else (o/'ablations').resolve())
PY
)

python scripts/12_run_ablations.py \
  "$MODEL_KEY" \
  "$OUT_ROOT/ablation_document_ids.txt" \
  --config "$OUT_ROOT/ablation_config.json"

echo "Ablation translations saved under: $OUT_ROOT/translations_controlled"
