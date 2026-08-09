from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from common import load_config, load_evaluation_rows, resolve, stable_fraction, write_csv, write_json

METRICS = ["semantic_fidelity", "legal_accuracy", "terminology_compliance", "fluency", "overall_quality"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = Counter(str((row.get("judgment") or {}).get("winner", "tie")) for row in rows)
    result: dict[str, Any] = {
        "paragraphs": len(rows),
        "wins": {"baseline": wins["baseline"], "controlled": wins["controlled"], "tie": wins["tie"]},
        "controlled_win_rate": wins["controlled"] / len(rows) if rows else None,
        "decisive_controlled_win_rate": wins["controlled"] / (wins["baseline"] + wins["controlled"]) if wins["baseline"] + wins["controlled"] else None,
        "score_deltas_controlled_minus_baseline": {},
    }
    for metric in METRICS:
        values = []
        for row in rows:
            scores = (row.get("judgment") or {}).get("scores") or {}
            try:
                values.append(float(scores["controlled"][metric]) - float(scores["baseline"][metric]))
            except (KeyError, TypeError, ValueError):
                pass
        result["score_deltas_controlled_minus_baseline"][metric] = sum(values) / len(values) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive the v1.8.1 fixed low-cost subset from already completed full evaluations without API calls.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--non-term-sample-rate", type=float, default=0.10)
    parser.add_argument("--sample-seed", type=int, default=2026)
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    output_root = resolve(root, paths["output_root"])
    method_dirs = {
        "glossary": resolve(root, paths["glossary_evaluation"]),
        "placeholder": resolve(root, paths["placeholder_evaluation"]),
    }
    loaded = {method: load_evaluation_rows(directory)[0] for method, directory in method_dirs.items()}
    common = sorted(set(loaded["glossary"]) & set(loaded["placeholder"]))
    selected_keys = []
    for document_id, paragraph_id in common:
        row = loaded["glossary"][(document_id, paragraph_id)]
        if row.get("has_benchmark_terms") or stable_fraction(args.sample_seed, document_id, paragraph_id) < args.non_term_sample_rate:
            selected_keys.append((document_id, paragraph_id))

    summary = {
        "sample_seed": args.sample_seed,
        "non_term_sample_rate": args.non_term_sample_rate,
        "common_full_paragraphs": len(common),
        "selected_common_paragraphs": len(selected_keys),
        "selected_term_bearing": sum(int(loaded["glossary"][key].get("has_benchmark_terms")) for key in selected_keys),
        "selected_non_term": sum(int(not loaded["glossary"][key].get("has_benchmark_terms")) for key in selected_keys),
        "methods": {},
    }
    rows_out = []
    for method in ("glossary", "placeholder"):
        rows = [loaded[method][key] for key in selected_keys]
        summary["methods"][method] = {
            "overall": summarize(rows),
            "term_bearing": summarize([row for row in rows if row.get("has_benchmark_terms")]),
            "non_term_bearing": summarize([row for row in rows if not row.get("has_benchmark_terms")]),
        }
        for subset, values in summary["methods"][method].items():
            rows_out.append({
                "method": method,
                "subset": subset,
                "paragraphs": values["paragraphs"],
                "controlled_win_rate": values["controlled_win_rate"],
                "decisive_controlled_win_rate": values["decisive_controlled_win_rate"],
                **{f"delta_{metric}": value for metric, value in values["score_deltas_controlled_minus_baseline"].items()},
            })

    write_json(output_root / "07_lowcost_subset_from_full.json", summary)
    write_csv(output_root / "07_lowcost_subset_from_full.csv", rows_out)
    write_csv(output_root / "07_lowcost_subset_keys.csv", [
        {"document_id": doc, "paragraph_id": pid, "has_benchmark_terms": loaded["glossary"][(doc, pid)].get("has_benchmark_terms")}
        for doc, pid in selected_keys
    ])
    print(f"Selected {len(selected_keys)} of {len(common)} common full-evaluation paragraphs")
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
