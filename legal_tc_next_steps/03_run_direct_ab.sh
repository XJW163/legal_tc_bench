#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-analysis_config.json}"

python - "$CONFIG" <<'PY' > /tmp/legal_tc_direct_ab.env
import json, shlex, sys
from pathlib import Path
cfg_path = Path(sys.argv[1]).resolve()
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
root = Path(cfg.get("project_root", "."))
if not root.is_absolute():
    root = (cfg_path.parent / root).resolve()
paths = cfg["paths"]
out = Path(paths["output_root"])
if not out.is_absolute(): out = root / out
sample = out / "direct_ab_sample"
eval_root = Path(paths["evaluation_root"])
if not eval_root.is_absolute(): eval_root = root / eval_root
judge = cfg["judge"]
exp = cfg["direct_ab"]["experiment"]
values = {
    "ANNOTATED": sample / "annotated",
    "GLOSSARY": sample / "glossary",
    "PLACEHOLDER": sample / "placeholder",
    "TERM_MEMORY": sample / "term_memory",
    "EVAL_ROOT": eval_root,
    "EXPERIMENT": exp,
    "MODEL": judge["model"],
    "BASE_URL": judge["base_url"],
    "WORKERS": judge.get("workers", 16),
    "MAX_TOKENS": judge.get("max_tokens", 1200),
    "MAX_RETRIES": judge.get("max_retries", 2),
    "MAX_ESCALATIONS": judge.get("max_token_escalations", 1),
    "TIMEOUT": judge.get("timeout", 300),
    "EXTRA_BODY": judge.get("extra_body_json", '{"thinking":{"type":"disabled"}}'),
    "LOG": root / "data/logs" / f"{exp}.log",
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
# shellcheck disable=SC1091
source /tmp/legal_tc_direct_ab.env
mkdir -p "$(dirname "$LOG")" "$EVAL_ROOT"

PYTHONUNBUFFERED=1 nohup legal-tc llm-evaluate-translation \
  "$ANNOTATED" \
  "$GLOSSARY" \
  "$PLACEHOLDER" \
  "$EVAL_ROOT" \
  --experiment "$EXPERIMENT" \
  --term-memory "$TERM_MEMORY" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --baseline-label glossary \
  --controlled-label placeholder \
  --position-control bidirectional \
  --evaluation-scope all \
  --extra-body-json "$EXTRA_BODY" \
  --max-tokens "$MAX_TOKENS" \
  --max-retries "$MAX_RETRIES" \
  --max-token-escalations "$MAX_ESCALATIONS" \
  --workers "$WORKERS" \
  --timeout "$TIMEOUT" \
  > "$LOG" 2>&1 &

PID=$!
echo "Started PID $PID"
echo "Log: $LOG"
echo "Follow: tail -f '$LOG' | tr '\\r' '\\n'"
