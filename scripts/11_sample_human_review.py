from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import experiment_registry, load_config, p, write_csv


RISK_STATUSES = {"OMITTED", "AMBIGUOUS", "TRANSLATION_ERROR", "REPAIR_FAILED_RETAINED"}


def iter_rows(directory: Path):
    for path in directory.rglob("*.json"):
        if path.name.endswith(("manifest.json", ".audit.json", ".checkpoint.json")):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = value if isinstance(value, list) else value.get("rows") or value.get("occurrences") or []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("occurrence_id"):
                    yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--risk-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config, root = load_config(args.config)
    clean_root = p(root, config["paths"]["renderings_clean_root"])
    results_root = p(root, config["paths"]["results_root"])
    pool = []
    for exp in experiment_registry(config, root, include_missing=False):
        directory = clean_root / exp["experiment_id"]
        for row in iter_rows(directory):
            status = str(row.get("validation_status") or row.get("extraction_status") or "")
            pool.append({
                "experiment_id": exp["experiment_id"],
                "model_key": exp["model_key"],
                "method": exp["method"],
                "occurrence_id": row.get("occurrence_id"),
                "document_id": row.get("document_id"),
                "term_id": row.get("term_id"),
                "source_term": row.get("source_term"),
                "criticality": row.get("criticality"),
                "source_context": row.get("source_context") or row.get("source_marked_context"),
                "target_paragraph": row.get("target_paragraph"),
                "rendering": row.get("rendering"),
                "automatic_status": status,
                "is_risk": status in RISK_STATUSES or row.get("include_in_consistency") is False,
            })

    rng = random.Random(args.seed)
    risky = [x for x in pool if x["is_risk"]]
    normal = [x for x in pool if not x["is_risk"]]
    rng.shuffle(risky)
    rng.shuffle(normal)
    risk_n = min(len(risky), round(args.sample_size * args.risk_fraction))
    sample = risky[:risk_n] + normal[: max(0, args.sample_size - risk_n)]
    rng.shuffle(sample)
    for index, row in enumerate(sample, 1):
        row["sample_id"] = f"H{index:04d}"
        row["human_label"] = ""
        row["term_semantics_correct"] = ""
        row["legal_distinction_preserved"] = ""
        row["meaning_complete"] = ""
        row["naturalness"] = ""
        row["reviewer_notes"] = ""
    write_csv(results_root / "human_review_sample.csv", sample)
    print(f"wrote {results_root / 'human_review_sample.csv'} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
