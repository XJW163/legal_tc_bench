from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_config, load_evaluation_rows, resolve, write_csv, write_json

METRICS = ["semantic_fidelity", "legal_accuracy", "terminology_compliance", "fluency", "overall_quality"]


def delta(row: dict[str, Any], metric: str) -> float | None:
    scores = (row.get("judgment") or {}).get("scores") or {}
    try:
        return float(scores["controlled"][metric]) - float(scores["baseline"][metric])
    except (KeyError, TypeError, ValueError):
        return None


def winner(row: dict[str, Any]) -> str:
    return str((row.get("judgment") or {}).get("winner", "tie"))


def critical_count(row: dict[str, Any], role: str) -> int:
    errors = ((row.get("judgment") or {}).get("critical_errors") or {}).get(role) or {}
    return sum(int(bool(value)) for value in errors.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ranked qualitative cases for paper error analysis.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--per-category", type=int, default=50)
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    output_root = resolve(root, paths["output_root"]) / "error_analysis"
    glossary = load_evaluation_rows(resolve(root, paths["glossary_evaluation"]))[0]
    placeholder = load_evaluation_rows(resolve(root, paths["placeholder_evaluation"]))[0]
    common = sorted(set(glossary) & set(placeholder))

    all_rows = []
    for document_id, paragraph_id in common:
        g, p = glossary[(document_id, paragraph_id)], placeholder[(document_id, paragraph_id)]
        metric_values = {}
        for metric in METRICS:
            gd, pd = delta(g, metric), delta(p, metric)
            metric_values[f"glossary_delta_{metric}"] = gd
            metric_values[f"placeholder_delta_{metric}"] = pd
            metric_values[f"glossary_advantage_{metric}"] = gd - pd if gd is not None and pd is not None else None
        all_rows.append({
            "document_id": document_id,
            "paragraph_id": paragraph_id,
            "has_benchmark_terms": bool(g.get("has_benchmark_terms")),
            "source": g.get("source", ""),
            "baseline_translation": g.get("baseline_translation", ""),
            "glossary_translation": g.get("controlled_translation", ""),
            "placeholder_translation": p.get("controlled_translation", ""),
            "relevant_terms": json.dumps(g.get("relevant_terms") or [], ensure_ascii=False),
            "glossary_winner_vs_baseline": winner(g),
            "placeholder_winner_vs_baseline": winner(p),
            "glossary_rationale": (g.get("judgment") or {}).get("rationale", ""),
            "placeholder_rationale": (p.get("judgment") or {}).get("rationale", ""),
            "glossary_controlled_critical_errors": critical_count(g, "controlled"),
            "placeholder_controlled_critical_errors": critical_count(p, "controlled"),
            **metric_values,
        })

    categories = {
        "glossary_overall_advantage": (
            lambda row: row["glossary_advantage_overall_quality"] if row["has_benchmark_terms"] else None,
            True,
        ),
        "placeholder_overall_advantage": (
            lambda row: row["glossary_advantage_overall_quality"] if row["has_benchmark_terms"] else None,
            False,
        ),
        "glossary_terminology_advantage": (
            lambda row: row["glossary_advantage_terminology_compliance"] if row["has_benchmark_terms"] else None,
            True,
        ),
        "placeholder_fluency_harm": (
            lambda row: (
                row["glossary_delta_fluency"] - row["placeholder_delta_fluency"]
                if row["has_benchmark_terms"]
                and row["placeholder_delta_fluency"] is not None
                and row["placeholder_delta_fluency"] < 0
                else None
            ),
            True,
        ),
        "placeholder_critical_errors": (
            lambda row: float(row["placeholder_controlled_critical_errors"] - row["glossary_controlled_critical_errors"]),
            True,
        ),
    }

    selected_rows = []
    category_manifest = {}
    for category, (score_fn, descending) in categories.items():
        candidates = []
        for row in all_rows:
            score = score_fn(row)
            if score is None:
                continue
            candidates.append((float(score), row))
        candidates.sort(key=lambda item: (item[0], item[1]["document_id"], str(item[1]["paragraph_id"])), reverse=descending)
        if category == "placeholder_overall_advantage":
            candidates = [item for item in candidates if item[0] < 0]
        else:
            candidates = [item for item in candidates if item[0] > 0]
        chosen = candidates[: args.per_category]
        category_manifest[category] = len(chosen)
        for rank, (score, row) in enumerate(chosen, 1):
            selected_rows.append({"category": category, "rank": rank, "category_score": score, **row})

    write_csv(output_root / "all_paired_cases_rankable.csv", all_rows)
    write_csv(output_root / "selected_qualitative_cases.csv", selected_rows)
    write_json(output_root / "error_case_manifest.json", {
        "common_paragraphs": len(common),
        "per_category_requested": args.per_category,
        "selected_by_category": category_manifest,
    })
    print(f"Exported {len(selected_rows)} selected cases under: {output_root}")


if __name__ == "__main__":
    main()
