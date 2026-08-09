#!/usr/bin/env bash
set -euo pipefail
ANALYSIS_CONFIG="${1:-analysis_config.json}"
CONFIG=$(python legal_tc_next_steps/01b_make_semantic_metrics_config.py --config "$ANALYSIS_CONFIG")
echo "Using config: $CONFIG"
python scripts/04_build_reviews.py --config "$CONFIG"
python scripts/05_extract_renderings.py --config "$CONFIG"
python scripts/06_validate_and_repair_renderings.py --config "$CONFIG"
python scripts/07_aggregate_judge_metrics.py --config "$CONFIG"
echo "Semantic metrics complete."
