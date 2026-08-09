from __future__ import annotations

import argparse
from pathlib import Path

from common import benchmark_terms, index_documents, load_config, load_json, resolve, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a representative document subset for closed-loop ablations.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--documents", type=int, default=20)
    args = parser.parse_args()

    config, root = load_config(args.config)
    annotated_index = index_documents(resolve(root, config["paths"]["annotated"]), "paragraphs")
    output_root = resolve(root, config["paths"]["output_root"]) / "ablations"

    rows = []
    for document_id, path in sorted(annotated_index.items()):
        data = load_json(path)
        term_pids = set()
        occurrence_count = 0
        for term in benchmark_terms(data):
            for occurrence in term.get("occurrences") or []:
                if isinstance(occurrence, dict) and "paragraph_id" in occurrence:
                    term_pids.add(str(occurrence.get("paragraph_id")))
                    occurrence_count += 1
        rows.append({
            "document_id": document_id,
            "paragraphs": len(data.get("paragraphs") or []),
            "term_bearing_paragraphs": len(term_pids),
            "term_occurrences": occurrence_count,
        })

    ranked = sorted(rows, key=lambda row: (row["term_bearing_paragraphs"], row["term_occurrences"], row["document_id"]))
    n = min(args.documents, len(ranked))
    if n == 0:
        raise ValueError("no annotated documents found")
    # Evenly spaced ranks cover low-, medium-, and high-complexity documents.
    indices = sorted(set(round(i * (len(ranked) - 1) / max(1, n - 1)) for i in range(n)))
    selected = [ranked[i] for i in indices]
    while len(selected) < n:
        for row in reversed(ranked):
            if row not in selected:
                selected.append(row)
                if len(selected) == n:
                    break
    selected.sort(key=lambda row: row["document_id"])

    output_root.mkdir(parents=True, exist_ok=True)
    ids_path = output_root / "ablation_document_ids.txt"
    ids_path.write_text("\n".join(row["document_id"] for row in selected) + "\n", encoding="utf-8")
    write_csv(output_root / "ablation_document_statistics.csv", selected)
    write_json(output_root / "ablation_selection.json", {
        "selection": "evenly spaced by term-bearing paragraph count",
        "requested_documents": args.documents,
        "selected_documents": len(selected),
        "documents": selected,
    })
    print(f"Selected {len(selected)} documents: {ids_path}")


if __name__ == "__main__":
    main()
