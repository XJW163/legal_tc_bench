#!/usr/bin/env bash
set -euo pipefail

cd /home/xujiwei/legal_tc_bench

BASE_URL="http://127.0.0.1:8001/v1"
MODEL_KEY="aya-expanse-32b"
MODE="${1:-smoke}"
case "$MODE" in
  smoke) DOCUMENT_LIMIT=(--max-documents 1) ;;
  final) DOCUMENT_LIMIT=() ;;
  *) echo "usage: $0 [smoke|final]" >&2; exit 2 ;;
esac
LOG="data/logs/${MODEL_KEY}-controlled-${MODE}.log"
mkdir -p data/logs data/translations_controlled

for path in \
  run_translation_unicode_recovery_v2.py \
  run_controlled_with_nonlinguistic_passthrough.py \
  data/annotated \
  data/term_memory_frozen_v1 \
  data/occurrences_validated_all
do
  test -e "$path" || { echo "missing: $path" >&2; exit 1; }
done

MODEL_ID="$(
  curl -fsS "$BASE_URL/models" \
  | jq -er 'if (.data | length) == 1 then .data[0].id else error("endpoint must advertise exactly one model") end'
)"

echo "endpoint: $BASE_URL"
echo "model:    $MODEL_ID"
curl -fsS "$BASE_URL/models" | jq '.data[] | {id, max_model_len}'

LEGAL_TC_RUNNER_MODULE=run_translation_unicode_recovery_v2 \
PYTHONUNBUFFERED=1 python3 \
  run_controlled_with_nonlinguistic_passthrough.py \
  --model "$MODEL_ID" \
  --base-url "$BASE_URL" \
  --model-key "$MODEL_KEY" \
  --methods glossary placeholder \
  --annotated data/annotated \
  --term-memory data/term_memory_frozen_v1 \
  --occurrence-validation data/occurrences_validated_all \
  --output-root data/translations_controlled \
  --experiment-suffix "$MODE" \
  --temperature 0 \
  --max-tokens 4096 \
  --context-paragraphs 2 \
  --max-context-chars 12000 \
  --max-retries 5 \
  --timeout 600 \
  --workers 16 \
  --normalize-unicode-spaces \
  "${DOCUMENT_LIMIT[@]}" \
  2>&1 | tee "$LOG"

