from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import (
    load_config,
    load_evaluation_rows,
    mean,
    read_csv,
    resolve,
    write_csv,
    write_json,
)

METRICS = [
    "semantic_fidelity",
    "legal_accuracy",
    "terminology_compliance",
    "fluency",
    "overall_quality",
]

SUBSETS = [
    "overall",
    "term_bearing",
    "non_term_bearing",
]


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile for a non-empty list."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def arithmetic_mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)


def delta(row: dict[str, Any], metric: str) -> float:
    """Controlled-method score minus baseline score for one evaluation row."""
    scores = row["judgment"]["scores"]
    return float(scores["controlled"][metric]) - float(scores["baseline"][metric])


def document_level_stats(
    values_by_doc: dict[str, list[float]],
    *,
    bootstrap_seed: int,
    permutation_seed: int,
    bootstraps: int,
    permutations: int,
) -> dict[str, Any]:
    """
    Compute one internally consistent document-level paired analysis.

    Statistical unit: document.
      1. Average paragraph-level paired differences within each document.
      2. Give every document equal weight for the reported effect.
      3. Resample document means for the bootstrap confidence interval.
      4. Apply sign flips to the same document means for the two-sided p-value.

    This avoids the previous mismatch where Effect/CI were paragraph-weighted
    but the p-value was computed from document-equal macro means.
    """
    if bootstraps <= 0:
        raise ValueError("bootstraps must be positive")
    if permutations <= 0:
        raise ValueError("permutations must be positive")

    documents = sorted(values_by_doc)
    if not documents:
        raise ValueError("no documents available for significance analysis")

    empty_documents = [doc for doc in documents if not values_by_doc[doc]]
    if empty_documents:
        raise ValueError(f"documents with no values: {empty_documents[:5]}")

    document_means = [
        arithmetic_mean(values_by_doc[document_id])
        for document_id in documents
    ]

    # Primary effect used by Effect, CI and p: every document has equal weight.
    document_macro_effect = arithmetic_mean(document_means)

    # Descriptive secondary quantity retained in CSV/JSON only.
    all_paragraph_values = [
        value
        for document_id in documents
        for value in values_by_doc[document_id]
    ]
    paragraph_weighted_effect = arithmetic_mean(all_paragraph_values)

    n_documents = len(document_means)

    bootstrap_rng = random.Random(bootstrap_seed)
    bootstrap_distribution: list[float] = []
    for _ in range(bootstraps):
        sampled_indices = [
            bootstrap_rng.randrange(n_documents)
            for _ in range(n_documents)
        ]
        sampled_effect = arithmetic_mean(
            document_means[index]
            for index in sampled_indices
        )
        bootstrap_distribution.append(sampled_effect)

    ci95_low = percentile(bootstrap_distribution, 0.025)
    ci95_high = percentile(bootstrap_distribution, 0.975)

    permutation_rng = random.Random(permutation_seed)
    observed_absolute = abs(document_macro_effect)
    extreme = 0

    for _ in range(permutations):
        randomized_effect = arithmetic_mean(
            value * (-1.0 if permutation_rng.random() < 0.5 else 1.0)
            for value in document_means
        )
        if abs(randomized_effect) >= observed_absolute:
            extreme += 1

    # Add-one correction prevents reporting an impossible p-value of exactly 0.
    p_value = (extreme + 1) / (permutations + 1)

    return {
        "document_macro_effect": document_macro_effect,
        "paragraph_weighted_effect": paragraph_weighted_effect,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p": p_value,
        "documents": n_documents,
        "paragraphs": len(all_paragraph_values),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def maybe_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def winner_counts(
    records: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
) -> Counter[str]:
    return Counter(
        str((records[key].get("judgment") or {}).get("winner", "tie"))
        for key in keys
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready tables from the completed gpt-oss-20b "
            "experiments using a consistent document-level significance test."
        )
    )
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    output_root = resolve(root, paths["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    glossary = load_evaluation_rows(
        resolve(root, paths["glossary_evaluation"])
    )[0]
    placeholder = load_evaluation_rows(
        resolve(root, paths["placeholder_evaluation"])
    )[0]

    common = sorted(set(glossary) & set(placeholder))
    if not common:
        raise RuntimeError(
            "Glossary and Placeholder have no common successful evaluation rows."
        )

    subset_keys: dict[str, list[tuple[str, str]]] = {
        "overall": common,
        "term_bearing": [
            key
            for key in common
            if glossary[key].get("has_benchmark_terms")
        ],
        "non_term_bearing": [
            key
            for key in common
            if not glossary[key].get("has_benchmark_terms")
        ],
    }

    main_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []

    for subset_index, subset in enumerate(SUBSETS):
        keys = subset_keys[subset]
        if not keys:
            raise RuntimeError(f"subset {subset!r} contains no common rows")

        # Descriptive paragraph-level results for each method versus Baseline.
        for method, records in (
            ("glossary", glossary),
            ("placeholder", placeholder),
        ):
            wins = winner_counts(records, keys)
            decisive = wins["baseline"] + wins["controlled"]

            row: dict[str, Any] = {
                "subset": subset,
                "method": method,
                "paragraphs": len(keys),
                "controlled_wins": wins["controlled"],
                "baseline_wins": wins["baseline"],
                "ties": wins["tie"],
                "controlled_win_rate": (
                    wins["controlled"] / len(keys)
                    if keys
                    else None
                ),
                "decisive_controlled_win_rate": (
                    wins["controlled"] / decisive
                    if decisive
                    else None
                ),
            }

            for metric in METRICS:
                row[f"delta_{metric}"] = mean(
                    delta(records[key], metric)
                    for key in keys
                )

            main_rows.append(row)

        # Paired Glossary-minus-Placeholder analysis at the document level.
        for metric_index, metric in enumerate(METRICS):
            values_by_document: dict[str, list[float]] = defaultdict(list)

            for key in keys:
                document_id = key[0]
                paired_difference = (
                    delta(glossary[key], metric)
                    - delta(placeholder[key], metric)
                )
                values_by_document[document_id].append(paired_difference)

            base_seed = args.seed + 100 * subset_index + metric_index
            stats = document_level_stats(
                values_by_document,
                bootstrap_seed=base_seed,
                permutation_seed=base_seed + 1_000_000,
                bootstraps=args.bootstrap_samples,
                permutations=args.permutations,
            )

            significance_rows.append({
                "subset": subset,
                "metric": metric,
                "paragraphs": len(keys),
                "documents": stats["documents"],
                "document_macro_effect": stats["document_macro_effect"],
                "paragraph_weighted_effect": stats["paragraph_weighted_effect"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "document_sign_flip_p": stats["p"],
                "significant_at_0_05": stats["p"] < 0.05,
            })

    main_csv = output_root / "11_main_llm_judge_results.csv"
    significance_csv = (
        output_root /
        "11_glossary_vs_placeholder_significance.csv"
    )
    json_path = output_root / "11_paper_table_data.json"

    write_csv(main_csv, main_rows)
    write_csv(significance_csv, significance_rows)
    write_json(json_path, {
        "statistical_unit": "document",
        "effect_definition": (
            "Mean of within-document paragraph-level paired differences; "
            "every document has equal weight."
        ),
        "confidence_interval": (
            "95% percentile bootstrap interval obtained by resampling documents."
        ),
        "p_value": (
            "Two-sided document-level sign-flip randomization test with add-one correction."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "common_successful_paragraphs": len(common),
        "main_results": main_rows,
        "significance": significance_rows,
    })

    markdown: list[str] = [
        "# LegalTC experiment results",
        "",
        f"Paired common successful paragraphs: **{len(common)}**",
        "",
        "## LLM-judge results",
    ]

    descriptive_rows: list[list[Any]] = []
    for row in main_rows:
        descriptive_rows.append([
            row["subset"],
            row["method"],
            row["paragraphs"],
            fmt(row["decisive_controlled_win_rate"]),
            fmt(row["delta_semantic_fidelity"]),
            fmt(row["delta_legal_accuracy"]),
            fmt(row["delta_terminology_compliance"]),
            fmt(row["delta_fluency"]),
            fmt(row["delta_overall_quality"]),
        ])

    markdown.append(md_table(
        [
            "Subset",
            "Method",
            "N",
            "Decisive win",
            "ΔSem",
            "ΔLegal",
            "ΔTerm",
            "ΔFluency",
            "ΔOverall",
        ],
        descriptive_rows,
    ))

    markdown.extend([
        "",
        "## Glossary advantage over Placeholder",
        "",
        (
            "Effect, confidence interval, and p-value below all use the same "
            "document-equal statistic."
        ),
    ])

    significance_md_rows = [
        [
            row["subset"],
            row["metric"],
            row["documents"],
            fmt(row["document_macro_effect"], 6),
            (
                f"[{fmt(row['ci95_low'], 6)}, "
                f"{fmt(row['ci95_high'], 6)}]"
            ),
            fmt(row["document_sign_flip_p"], 6),
        ]
        for row in significance_rows
    ]

    markdown.append(md_table(
        [
            "Subset",
            "Metric",
            "Documents",
            "Effect (doc-macro)",
            "95% CI",
            "p",
        ],
        significance_md_rows,
    ))

    deterministic = maybe_csv(
        output_root / "01_deterministic_term_metrics.csv"
    )
    if deterministic:
        markdown.extend([
            "",
            "## Deterministic string-presence metrics",
        ])
        markdown.append(md_table(
            [
                "Method",
                "Canonical",
                "Acceptable",
                "Strict term",
                "Strict document",
            ],
            [
                [
                    row.get("method"),
                    fmt(row.get("canonical_presence_micro")),
                    fmt(row.get("acceptable_presence_micro")),
                    fmt(row.get("strict_acceptable_term_rate")),
                    fmt(row.get("strict_acceptable_document_rate")),
                ]
                for row in deterministic
            ],
        ))

    # Optional direct A/B results, only included when the file exists.
    direct = maybe_csv(output_root / "04_direct_ab_significance.csv")
    if direct:
        markdown.extend([
            "",
            "## Direct Glossary-vs-Placeholder blind comparison",
        ])
        markdown.append(md_table(
            [
                "Subset",
                "Metric",
                "Glossary−Placeholder",
                "95% CI",
                "p",
            ],
            [
                [
                    row.get("subset"),
                    row.get("metric"),
                    fmt(row.get("effect_glossary_minus_placeholder"), 6),
                    (
                        f"[{fmt(row.get('ci95_low'), 6)}, "
                        f"{fmt(row.get('ci95_high'), 6)}]"
                    ),
                    fmt(row.get("sign_flip_p"), 6),
                ]
                for row in direct
            ],
        ))

    # Optional human-review section; absent files are silently ignored.
    human_path = (
        output_root /
        "human_review" /
        "06_human_review_summary.json"
    )
    if human_path.exists():
        human = json.loads(human_path.read_text(encoding="utf-8"))
        markdown.extend([
            "",
            "## Human validation",
            "",
            f"- Raw reviewer agreement: {fmt(human.get('raw_reviewer_agreement'))}",
            f"- Cohen's κ: {fmt(human.get('cohens_kappa_preference'))}",
            (
                "- Human-consensus/LLM agreement: "
                f"{fmt(human.get('human_consensus_llm_agreement'))}"
            ),
        ])

    efficiency = maybe_csv(output_root / "10_efficiency_cost_table.csv")
    if efficiency:
        markdown.extend([
            "",
            "## Efficiency and cost",
        ])
        markdown.append(md_table(
            [
                "Method",
                "Judge tokens",
                "Cost (CNY)",
                "Tokens/paragraph",
                "Translation retries",
                "Fallbacks",
            ],
            [
                [
                    row.get("method"),
                    row.get("judge_total_tokens", "—"),
                    fmt(row.get("judge_actual_or_estimated_cost_cny"), 2),
                    fmt(row.get("judge_tokens_per_successful_paragraph"), 1),
                    row.get("generation_retries"),
                    row.get("fallback_paragraphs"),
                ]
                for row in efficiency
            ],
        ))

    report_path = output_root / "11_paper_results.md"
    report_path.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    print(f"Common successful paragraphs: {len(common)}")
    print(f"Saved: {main_csv}")
    print(f"Saved: {significance_csv}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
