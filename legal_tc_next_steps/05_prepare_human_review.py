from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_config, load_evaluation_rows, resolve, stable_order_key, write_csv, write_json

REVIEW_FIELDS = [
    "preferred",
    "term_correct_A",
    "term_correct_B",
    "legal_accuracy_A",
    "legal_accuracy_B",
    "fluency_A",
    "fluency_B",
    "critical_error_A",
    "critical_error_B",
    "reviewer_notes",
]


def mapped_winner(row: dict[str, Any]) -> str:
    winner = str((row.get("judgment") or {}).get("winner", "tie"))
    if winner == "baseline":
        return "glossary"
    if winner == "controlled":
        return "placeholder"
    return "tie"


def balanced_select(items: list[tuple[tuple[str, str], dict[str, Any]]], n: int, seed: int) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    groups: dict[str, list[tuple[tuple[str, str], dict[str, Any]]]] = defaultdict(list)
    for item in items:
        groups[mapped_winner(item[1])].append(item)
    for label, group in groups.items():
        group.sort(key=lambda item: stable_order_key(seed + len(label), item[0][0], item[0][1]))

    selected: list[tuple[tuple[str, str], dict[str, Any]]] = []
    labels = ["glossary", "placeholder", "tie"]
    base = n // len(labels)
    for label in labels:
        selected.extend(groups.get(label, [])[:base])

    selected_keys = {key for key, _ in selected}
    remaining = [item for item in items if item[0] not in selected_keys]
    remaining.sort(key=lambda item: stable_order_key(seed + 999, item[0][0], item[0][1]))
    selected.extend(remaining[: max(0, n - len(selected))])
    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a blinded two-reviewer human validation sample.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--term-fraction", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config, root = load_config(args.config)
    settings = config.get("human_review") or {}
    sample_size = args.sample_size if args.sample_size is not None else int(settings.get("sample_size", 200))
    term_fraction = args.term_fraction if args.term_fraction is not None else float(settings.get("term_fraction", 1.0))
    seed = args.seed if args.seed is not None else int(settings.get("seed", 2026))
    if not 0.0 <= term_fraction <= 1.0:
        raise ValueError("term_fraction must be in [0, 1]")

    paths = config["paths"]
    experiment = config["direct_ab"]["experiment"]
    direct_dir = resolve(root, paths["evaluation_root"]) / experiment
    output_root = resolve(root, paths["output_root"]) / "human_review"
    rows, errors, _ = load_evaluation_rows(direct_dir)
    if not rows:
        raise ValueError(f"no successful direct A/B rows found in {direct_dir}")

    items = sorted(rows.items())
    term_items = [item for item in items if item[1].get("has_benchmark_terms")]
    non_term_items = [item for item in items if not item[1].get("has_benchmark_terms")]
    term_n = min(len(term_items), round(sample_size * term_fraction))
    non_term_n = min(len(non_term_items), sample_size - term_n)
    selected = balanced_select(term_items, term_n, seed) + balanced_select(non_term_items, non_term_n, seed + 10000)
    if len(selected) < sample_size:
        used = {key for key, _ in selected}
        rest = [item for item in items if item[0] not in used]
        rest.sort(key=lambda item: stable_order_key(seed + 20000, item[0][0], item[0][1]))
        selected.extend(rest[: sample_size - len(selected)])
    selected.sort(key=lambda item: stable_order_key(seed + 30000, item[0][0], item[0][1]))

    reviewer_rows = []
    key_rows = []
    for index, ((document_id, paragraph_id), row) in enumerate(selected, 1):
        sample_id = f"H{index:04d}"
        swapped = stable_order_key(seed + 40000, document_id, paragraph_id)[0] & 1
        glossary_text = str(row.get("baseline_translation", ""))
        placeholder_text = str(row.get("controlled_translation", ""))
        if swapped:
            text_a, method_a = placeholder_text, "placeholder"
            text_b, method_b = glossary_text, "glossary"
        else:
            text_a, method_a = glossary_text, "glossary"
            text_b, method_b = placeholder_text, "placeholder"
        terms = row.get("relevant_terms") or []
        terms_text = " | ".join(
            f"{term.get('source_term', '')} → {term.get('canonical_translation', '')}"
            for term in terms if isinstance(term, dict)
        )
        review = {
            "sample_id": sample_id,
            "document_id": document_id,
            "paragraph_id": paragraph_id,
            "has_benchmark_terms": bool(row.get("has_benchmark_terms")),
            "source": row.get("source", ""),
            "relevant_terms": terms_text,
            "translation_A": text_a,
            "translation_B": text_b,
            **{field: "" for field in REVIEW_FIELDS},
        }
        reviewer_rows.append(review)
        key_rows.append({
            "sample_id": sample_id,
            "document_id": document_id,
            "paragraph_id": paragraph_id,
            "method_A": method_a,
            "method_B": method_b,
            "llm_winner": mapped_winner(row),
            "llm_confidence": (row.get("judgment") or {}).get("confidence"),
            "has_benchmark_terms": bool(row.get("has_benchmark_terms")),
        })

    write_csv(output_root / "reviewer_1.csv", reviewer_rows)
    write_csv(output_root / "reviewer_2.csv", reviewer_rows)
    write_csv(output_root / "BLIND_KEY_DO_NOT_SHARE.csv", key_rows)
    write_json(output_root / "sample_manifest.json", {
        "source_experiment": experiment,
        "sample_size": len(reviewer_rows),
        "term_bearing": sum(int(row["has_benchmark_terms"]) for row in reviewer_rows),
        "non_term_bearing": sum(int(not row["has_benchmark_terms"]) for row in reviewer_rows),
        "seed": seed,
        "direct_evaluation_failures_excluded": len(errors),
        "instructions": {
            "preferred": "Fill A, B, or TIE.",
            "scores": "Fill integer 1-5 for term correctness, legal accuracy, and fluency.",
            "critical_error": "Fill 1 if present, otherwise 0.",
        },
    })
    print(f"Prepared {len(reviewer_rows)} blinded rows")
    print(f"Reviewer files: {output_root / 'reviewer_1.csv'} and {output_root / 'reviewer_2.csv'}")
    print(f"Keep the key private: {output_root / 'BLIND_KEY_DO_NOT_SHARE.csv'}")


if __name__ == "__main__":
    main()
