from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_config, load_evaluation_rows, mean, resolve, write_csv, write_json

METRICS = [
    "semantic_fidelity",
    "legal_accuracy",
    "terminology_compliance",
    "fluency",
    "overall_quality",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def cluster_inference(by_doc: dict[str, list[float]], *, seed: int, bootstraps: int, permutations: int) -> dict[str, float]:
    docs = sorted(by_doc)
    sums = [sum(by_doc[doc]) for doc in docs]
    counts = [len(by_doc[doc]) for doc in docs]
    means = [s / n for s, n in zip(sums, counts)]
    observed = sum(sums) / sum(counts)
    macro = sum(means) / len(means)
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstraps):
        sample = [rng.randrange(len(docs)) for _ in docs]
        boot.append(sum(sums[i] for i in sample) / sum(counts[i] for i in sample))
    extreme = 0
    for _ in range(permutations):
        statistic = sum(value * (-1.0 if rng.random() < 0.5 else 1.0) for value in means) / len(means)
        if abs(statistic) >= abs(macro):
            extreme += 1
    return {
        "effect_glossary_minus_placeholder": observed,
        "document_macro_effect": macro,
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        "sign_flip_p": (extreme + 1) / (permutations + 1),
        "documents": len(docs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the direct blind glossary-vs-placeholder experiment.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=50000)
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    experiment = config["direct_ab"]["experiment"]
    eval_dir = resolve(root, paths["evaluation_root"]) / experiment
    output_root = resolve(root, paths["output_root"])
    seed = int(config["direct_ab"].get("seed", 2026))

    rows, errors, documents = load_evaluation_rows(eval_dir)
    output_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "experiment": experiment,
        "documents": len(documents),
        "successful_paragraphs": len(rows),
        "failed_paragraphs": len(errors),
        "subsets": {},
    }

    subsets = {
        "overall": list(rows.items()),
        "term_bearing": [(key, row) for key, row in rows.items() if row.get("has_benchmark_terms")],
        "non_term_bearing": [(key, row) for key, row in rows.items() if not row.get("has_benchmark_terms")],
    }

    for subset, items in subsets.items():
        winners = {"glossary": 0, "placeholder": 0, "tie": 0}
        for _, row in items:
            winner = str((row.get("judgment") or {}).get("winner", "tie"))
            if winner == "baseline":
                winners["glossary"] += 1
            elif winner == "controlled":
                winners["placeholder"] += 1
            else:
                winners["tie"] += 1
        decisive = winners["glossary"] + winners["placeholder"]
        subset_summary = {
            "paragraphs": len(items),
            "wins": winners,
            "glossary_win_rate": winners["glossary"] / len(items) if items else None,
            "placeholder_win_rate": winners["placeholder"] / len(items) if items else None,
            "decisive_glossary_win_rate": winners["glossary"] / decisive if decisive else None,
            "metrics": {},
        }

        print("=" * 78)
        print(f"{subset}: {len(items)} paragraphs")
        print(
            f"wins glossary={winners['glossary']} placeholder={winners['placeholder']} tie={winners['tie']} "
            f"decisive_glossary_win_rate={subset_summary['decisive_glossary_win_rate']:.4f}"
            if decisive else "no decisive judgments"
        )

        for metric in METRICS:
            by_doc: dict[str, list[float]] = defaultdict(list)
            for (document_id, _), row in items:
                scores = (row.get("judgment") or {}).get("scores") or {}
                try:
                    # Direct experiment labels: baseline=glossary, controlled=placeholder.
                    diff = float(scores["baseline"][metric]) - float(scores["controlled"][metric])
                except (KeyError, TypeError, ValueError):
                    continue
                by_doc[document_id].append(diff)
            stats = cluster_inference(
                by_doc,
                seed=seed + METRICS.index(metric),
                bootstraps=args.bootstrap_samples,
                permutations=args.permutations,
            ) if by_doc else {}
            subset_summary["metrics"][metric] = stats
            output_rows.append({"subset": subset, "metric": metric, "paragraphs": len(items), **stats})
            if stats:
                print(
                    f"{metric:25s} effect={stats['effect_glossary_minus_placeholder']:+.6f} "
                    f"95% CI=[{stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f}] "
                    f"p={stats['sign_flip_p']:.6f}"
                )
        summary["subsets"][subset] = subset_summary

    write_json(output_root / "04_direct_ab_summary.json", summary)
    write_csv(output_root / "04_direct_ab_significance.csv", output_rows)
    write_csv(output_root / "04_direct_ab_failures.csv", errors)
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
