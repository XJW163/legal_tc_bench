#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

: "${DEEPSEEK_API_KEY:?Export DEEPSEEK_API_KEY before starting this script}"

command -v legal-tc >/dev/null 2>&1 || {
  echo "legal-tc is not available in PATH" >&2
  exit 127
}
command -v jq >/dev/null 2>&1 || {
  echo "jq is not available in PATH" >&2
  exit 127
}

BASELINE="data/translations/aya-expanse-32b-zero-english-v2"
OUTPUT_ROOT="data/translation_evaluations"
TERM_MEMORY="data/term_memory_frozen_v1"
ANNOTATED="data/annotated"

for required in "$BASELINE" "$TERM_MEMORY" "$ANNOTATED"; do
  if [[ ! -d "$required" ]]; then
    echo "missing required directory: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT" data/logs

run_evaluation() {
  local method="$1"
  local controlled="data/translations_controlled/aya-expanse-32b-${method}-final"
  local experiment="aya-expanse-32b-${method}-deepseek-v4-flash-lowcost"
  local log="data/logs/${experiment}.log"

  if [[ ! -d "$controlled" ]]; then
    echo "missing controlled directory: $controlled" >&2
    exit 2
  fi

  echo "===== ${experiment} ====="

  PYTHONUNBUFFERED=1 legal-tc llm-evaluate-translation \
    "$ANNOTATED" \
    "$BASELINE" \
    "$controlled" \
    "$OUTPUT_ROOT" \
    --experiment "$experiment" \
    --term-memory "$TERM_MEMORY" \
    --model deepseek-v4-flash \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --baseline-label baseline \
    --controlled-label "$method" \
    --temperature 0 \
    --position-control deterministic \
    --evaluation-scope term-bearing-plus-sample \
    --non-term-sample-rate 0.10 \
    --sample-seed 2026 \
    --extra-body-json '{"thinking":{"type":"disabled"}}' \
    --max-tokens 1200 \
    --max-token-escalations 1 \
    --workers 16 \
    --max-retries 2 \
    --timeout 300 \
    2>&1 | tee "$log"

  echo "${experiment} complete"
}

run_evaluation glossary
run_evaluation placeholder

echo "all Aya-Expanse-32B low-cost evaluations complete"
