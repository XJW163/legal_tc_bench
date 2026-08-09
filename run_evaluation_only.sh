#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/final_experiment.json}"
PYTHON="${PYTHON:-python}"

$PYTHON scripts/03_validate_experiments.py --config "$CONFIG"
$PYTHON scripts/04_build_reviews.py --config "$CONFIG"
$PYTHON scripts/05_extract_renderings.py --config "$CONFIG"
$PYTHON scripts/06_validate_and_repair_renderings.py --config "$CONFIG"
$PYTHON scripts/07_aggregate_judge_metrics.py --config "$CONFIG"
$PYTHON scripts/08_collect_reliability.py --config "$CONFIG"
$PYTHON scripts/09_statistical_analysis.py --config "$CONFIG"
$PYTHON scripts/10_make_paper_tables.py --config "$CONFIG"
$PYTHON scripts/11_sample_human_review.py --config "$CONFIG" --sample-size 100
