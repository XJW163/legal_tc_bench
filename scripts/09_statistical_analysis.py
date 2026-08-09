from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import atomic_json, load_config, load_json, p, write_csv


def document_metrics(metrics_file: Path) -> dict[str, dict[str, float]]:
    data = load_json(metrics_file, {}) or {}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("terms", []):
        by_doc[str(row.get("document_id"))].append(row)
    result: dict[str, dict[str, float]] = {}
    for doc, rows in by_doc.items():
        dtrs = [float(r["dominant_translation_ratio"]) for r in rows if isinstance(r.get("dominant_translation_ratio"), (int, float))]
        exact = sum(int(r.get("exact_occurrences", 0) or 0) for r in rows)
        acceptable = sum(int(r.get("acceptable_variant_occurrences", 0) or 0) for r in rows)
        inconsistent = sum(int(r.get("legal_inconsistency_occurrences", 0) or 0) for r in rows)
        omitted = sum(int(r.get("omitted_occurrences", 0) or 0) for r in rows)
        total = sum(int(r.get("total_occurrences", 0) or 0) for r in rows)
        semantic_den = exact + acceptable + inconsistent
        result[doc] = {
            "dtr_macro": mean(dtrs) if dtrs else math.nan,
            "slcr": (exact + acceptable) / semantic_den if semantic_den else math.nan,
            "lir": inconsistent / semantic_den if semantic_den else math.nan,
            "omission_rate": omitted / total if total else math.nan,
        }
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap_ci(diffs: list[float], *, seed: int = 42, samples: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return math.nan, math.nan
    means = [mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def sign_test_p(diffs: list[float]) -> float:
    nonzero = [x for x in diffs if x != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(1 for x in nonzero if x > 0)
    tail = min(positive, n - positive)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2 * probability)


def wilcoxon_p(diffs: list[float]) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    nonzero = [x for x in diffs if x != 0]
    if not nonzero:
        return 1.0
    try:
        return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    config, root = load_config(args.config)
    metrics_root = p(root, config["paths"]["final_metrics_root"])
    results_root = p(root, config["paths"]["results_root"])
    rows = []
    detailed = []
    for model in config["models"]:
        key = model["key"]
        files = {
            method: metrics_root / f"{key}__{method}.metrics.json"
            for method in ("baseline", "glossary", "placeholder")
        }
        if not all(path.exists() for path in files.values()):
            rows.append({"model_key": key, "status": "missing_metrics"})
            continue
        docs = {method: document_metrics(path) for method, path in files.items()}
        for comparator in ("glossary", "placeholder"):
            common_docs = sorted(set(docs["baseline"]) & set(docs[comparator]))
            for metric in ("dtr_macro", "slcr", "lir", "omission_rate"):
                pairs = []
                for doc in common_docs:
                    before = docs["baseline"][doc][metric]
                    after = docs[comparator][doc][metric]
                    if math.isnan(before) or math.isnan(after):
                        continue
                    diff = after - before
                    pairs.append((doc, before, after, diff))
                diffs = [x[3] for x in pairs]
                ci_low, ci_high = bootstrap_ci(diffs, samples=args.bootstrap_samples)
                direction_multiplier = -1.0 if metric in {"lir", "omission_rate"} else 1.0
                improved = sum(1 for x in diffs if direction_multiplier * x > 0)
                row = {
                    "model_key": key,
                    "comparison": f"{comparator}_vs_baseline",
                    "metric": metric,
                    "paired_documents": len(diffs),
                    "baseline_mean": mean(x[1] for x in pairs) if pairs else None,
                    "comparison_mean": mean(x[2] for x in pairs) if pairs else None,
                    "mean_difference": mean(diffs) if diffs else None,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "improved_documents": improved,
                    "improved_fraction": improved / len(diffs) if diffs else None,
                    "wilcoxon_p": wilcoxon_p(diffs),
                    "sign_test_p": sign_test_p(diffs),
                    "status": "ok",
                }
                rows.append(row)
                detailed.extend({
                    "model_key": key,
                    "comparison": f"{comparator}_vs_baseline",
                    "metric": metric,
                    "document_id": doc,
                    "baseline": before,
                    "comparison_value": after,
                    "difference": diff,
                } for doc, before, after, diff in pairs)

    write_csv(results_root / "paired_statistics.csv", rows)
    write_csv(results_root / "paired_document_differences.csv", detailed)
    atomic_json(results_root / "paired_statistics.json", {"comparisons": rows})
    print(f"wrote {results_root / 'paired_statistics.csv'}")


if __name__ == "__main__":
    main()
