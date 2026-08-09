#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/final_experiment.json}"
PYTHON="${PYTHON:-python}"

$PYTHON scripts/00_preflight.py --config "$CONFIG"

cat <<'EOF'
Translation is model-server dependent. Load each model and run, for example:
  python scripts/01_run_model.py qwen3-8b --config config/final_experiment.json
Repeat for all seven model keys. Then continue with:
  ./run_evaluation_only.sh config/final_experiment.json
EOF
