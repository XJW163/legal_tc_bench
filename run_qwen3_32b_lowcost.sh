#!/usr/bin/env bash
set -euo pipefail

# PROJECT_ROOT="${PROJECT_ROOT:-/home/xujiwei/legal_tc_bench}"
# cd "$PROJECT_ROOT"

mkdir -p data/logs data/translation_evaluations

for path in \
  data/annotated \
  data/term_memory_frozen_v1 \
  data/translations/qwen3-32b-zero-english-v3 \
  data/translations_controlled/qwen3-32b-glossary-final \
  data/translations_controlled/qwen3-32b-placeholder-final
do
  test -e "$path" || { echo "missing: $path" >&2; exit 1; }
done

run_evaluation() {
  local method="$1"
  local experiment="qwen3-32b-${method}-deepseek-v4-flash-lowcost"
  local controlled_dir="data/translations_controlled/qwen3-32b-${method}-final"
  local log="data/logs/${experiment}.log"

  echo "===== ${experiment} ====="

  PYTHONUNBUFFERED=1 legal-tc llm-evaluate-translation \
    data/annotated \
    data/translations/qwen3-32b-zero-english-v3 \
    "$controlled_dir" \
    data/translation_evaluations \
    --experiment "$experiment" \
    --term-memory data/term_memory_frozen_v1 \
    --model deepseek-v4-flash \
    --base-url https://api.deepseek.com \
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

echo "all Qwen3-32B low-cost evaluations complete"
