from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_config, mean, read_csv, resolve, write_csv, write_json

LABELS = ["glossary", "placeholder", "tie"]
SCORE_BASES = ["term_correct", "legal_accuracy", "fluency"]


def normalize_preferred(value: str) -> str | None:
    value = str(value or "").strip().upper()
    return value if value in {"A", "B", "TIE"} else None


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def map_preference(preferred: str | None, key: dict[str, str]) -> str | None:
    if preferred is None:
        return None
    if preferred == "TIE":
        return "tie"
    return key[f"method_{preferred}"]


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float | None, float | None]:
    pairs = [(x, y) for x, y in zip(a, b) if x in LABELS and y in LABELS]
    if not pairs:
        return None, None
    observed = sum(x == y for x, y in pairs) / len(pairs)
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum((ca[label] / len(pairs)) * (cb[label] / len(pairs)) for label in LABELS)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return observed, kappa


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze two completed blinded human-review CSV files.")
    parser.add_argument("reviewer_1", type=Path)
    parser.add_argument("reviewer_2", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--config", default="analysis_config.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    output_root = resolve(root, config["paths"]["output_root"]) / "human_review"
    key_path = args.key or (output_root / "BLIND_KEY_DO_NOT_SHARE.csv")
    reviewer1 = {row["sample_id"]: row for row in read_csv(args.reviewer_1)}
    reviewer2 = {row["sample_id"]: row for row in read_csv(args.reviewer_2)}
    keys = {row["sample_id"]: row for row in read_csv(key_path)}

    item_rows: list[dict[str, Any]] = []
    labels_1, labels_2 = [], []
    llm_labels, consensus_labels = [], []
    score_differences: dict[str, dict[str, list[float]]] = {
        "reviewer_1": {name: [] for name in SCORE_BASES},
        "reviewer_2": {name: [] for name in SCORE_BASES},
    }

    for sample_id in sorted(set(keys) & set(reviewer1) & set(reviewer2)):
        key = keys[sample_id]
        r1, r2 = reviewer1[sample_id], reviewer2[sample_id]
        label1 = map_preference(normalize_preferred(r1.get("preferred", "")), key)
        label2 = map_preference(normalize_preferred(r2.get("preferred", "")), key)
        labels_1.append(label1 or "missing")
        labels_2.append(label2 or "missing")
        consensus = label1 if label1 == label2 and label1 is not None else "disagreement"
        llm = key.get("llm_winner", "tie")
        if consensus != "disagreement":
            consensus_labels.append(consensus)
            llm_labels.append(llm)

        item = {
            "sample_id": sample_id,
            "document_id": key.get("document_id"),
            "paragraph_id": key.get("paragraph_id"),
            "reviewer_1_preference": label1,
            "reviewer_2_preference": label2,
            "consensus": consensus,
            "llm_winner": llm,
            "reviewers_agree": label1 == label2 and label1 is not None,
            "consensus_matches_llm": consensus == llm if consensus != "disagreement" else None,
        }

        for reviewer_name, row in (("reviewer_1", r1), ("reviewer_2", r2)):
            for base in SCORE_BASES:
                score_a = to_float(row.get(f"{base}_A", ""))
                score_b = to_float(row.get(f"{base}_B", ""))
                if score_a is None or score_b is None:
                    continue
                method_a, method_b = key["method_A"], key["method_B"]
                glossary_score = score_a if method_a == "glossary" else score_b
                placeholder_score = score_a if method_a == "placeholder" else score_b
                diff = glossary_score - placeholder_score
                score_differences[reviewer_name][base].append(diff)
                item[f"{reviewer_name}_{base}_glossary_minus_placeholder"] = diff
        item_rows.append(item)

    observed, kappa = cohens_kappa(labels_1, labels_2)
    consensus_count = len(consensus_labels)
    llm_agreement = (
        sum(h == l for h, l in zip(consensus_labels, llm_labels)) / consensus_count
        if consensus_count else None
    )
    consensus_counts = Counter(consensus_labels)
    decisive_g = consensus_counts["glossary"]
    decisive_p = consensus_counts["placeholder"]

    summary = {
        "samples_in_both_files": len(item_rows),
        "raw_reviewer_agreement": observed,
        "cohens_kappa_preference": kappa,
        "consensus_samples": consensus_count,
        "reviewer_disagreements": len(item_rows) - consensus_count,
        "consensus_counts": dict(consensus_counts),
        "consensus_glossary_decisive_rate": decisive_g / (decisive_g + decisive_p) if decisive_g + decisive_p else None,
        "consensus_glossary_vs_placeholder_sign_test_p": exact_sign_p(decisive_g, decisive_p),
        "human_consensus_llm_agreement": llm_agreement,
        "mean_score_difference_glossary_minus_placeholder": {
            reviewer: {metric: mean(values) for metric, values in metrics.items()}
            for reviewer, metrics in score_differences.items()
        },
    }

    write_json(output_root / "06_human_review_summary.json", summary)
    write_csv(output_root / "06_human_review_items.csv", item_rows)
    print(f"Samples: {len(item_rows)}")
    print(f"Reviewer agreement: {observed:.4f}" if observed is not None else "Reviewer agreement: N/A")
    print(f"Cohen's kappa: {kappa:.4f}" if kappa is not None else "Cohen's kappa: N/A")
    print(f"Consensus vs LLM agreement: {llm_agreement:.4f}" if llm_agreement is not None else "Consensus vs LLM agreement: N/A")
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
