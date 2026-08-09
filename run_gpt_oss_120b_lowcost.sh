#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xujiwei/legal_tc_bench}"
cd "$PROJECT_ROOT"

mkdir -p data/logs data/translation_evaluations

run_evaluation() {
  local method="$1"
  local controlled_label="$2"
  local controlled_dir="data/translations_controlled/gpt-oss-120b-${method}-final"
  local experiment="gpt-oss-120b-${method}-deepseek-v4-flash-lowcost"
  local log="data/logs/${experiment}.log"

  echo "===== ${experiment} ====="

  PYTHONUNBUFFERED=1 legal-tc llm-evaluate-translation \
    data/annotated \
    data/translations/gpt-oss-120b-zero-english-v2 \
    "$controlled_dir" \
    data/translation_evaluations \
    --experiment "$experiment" \
    --term-memory data/term_memory_frozen_v1 \
    --model deepseek-v4-flash \
    --base-url https://api.deepseek.com \
    --baseline-label baseline \
    --controlled-label "$controlled_label" \
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

run_evaluation glossary glossary
run_evaluation placeholder placeholder

echo "all gpt-oss-120b low-cost evaluations complete"
