from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import index_documents, load_config, load_evaluation_rows, load_json, mean, resolve, write_csv, write_json


def usage_total(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage") or {}
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def translation_stats(directory: Path) -> dict[str, Any]:
    index = index_documents(directory, "translations")
    translations = []
    doc_statistics = []
    for path in index.values():
        data = load_json(path)
        translations.extend(row for row in (data.get("translations") or []) if isinstance(row, dict))
        if isinstance(data.get("statistics"), dict):
            doc_statistics.append(data["statistics"])
    strategies = Counter(str(row.get("constraint_strategy_used") or "unknown") for row in translations)
    return {
        "documents": len(index),
        "paragraphs": len(translations),
        "latency_seconds_sum": sum(float(row.get("latency_seconds", 0) or 0) for row in translations),
        "latency_seconds_mean": mean(float(row.get("latency_seconds", 0) or 0) for row in translations),
        "attempts_mean": mean(float(row.get("attempts", 0) or 0) for row in translations),
        "generation_retries": sum(int(row.get("generation_retry_count", row.get("placeholder_retry_count", 0)) or 0) for row in translations),
        "glossary_compliance_retries": sum(int(row.get("glossary_compliance_retry_count", 0) or 0) for row in translations),
        "semantic_verifier_calls": sum(int(row.get("placeholder_semantic_verifier_calls", 0) or 0) for row in translations),
        "fallback_paragraphs": sum(int(bool(row.get("placeholder_fallback_used"))) for row in translations),
        "unresolved_paragraphs": sum(int(str(row.get("resolution_status")) in {"glossary_fallback_noncompliant", "noncompliant"}) for row in translations),
        "strategy_counts": dict(strategies),
    }


def evaluation_stats(directory: Path) -> dict[str, Any]:
    rows_map, errors, documents = load_evaluation_rows(directory)
    rows = list(rows_map.values())
    usage = usage_total(rows + errors)
    return {
        "documents": len(documents),
        "paragraphs_successful": len(rows),
        "paragraphs_failed": len(errors),
        **usage,
        "judge_latency_seconds_sum": sum(float(row.get("latency_seconds", 0) or 0) for row in rows),
        "judge_latency_seconds_mean": mean(float(row.get("latency_seconds", 0) or 0) for row in rows),
        "judge_attempts_sum": sum(int(row.get("attempts", 0) or 0) for row in rows),
        "judge_attempts_mean": mean(float(row.get("attempts", 0) or 0) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translation/evaluation efficiency and cost tables.")
    parser.add_argument("--config", default="analysis_config.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    output_root = resolve(root, paths["output_root"])
    costs = config.get("costs_cny") or {}

    eval_dirs = {
        "glossary": resolve(root, paths["glossary_evaluation"]),
        "placeholder": resolve(root, paths["placeholder_evaluation"]),
    }
    translation_dirs = {
        "baseline": resolve(root, paths["baseline"]),
        "glossary": resolve(root, paths["glossary"]),
        "placeholder": resolve(root, paths["placeholder"]),
    }
    eval_stats = {method: evaluation_stats(path) for method, path in eval_dirs.items()}
    translation = {method: translation_stats(path) for method, path in translation_dirs.items()}

    actual_glossary = costs.get("glossary_judge_actual")
    observed_rate = (
        float(actual_glossary) / eval_stats["glossary"]["total_tokens"]
        if actual_glossary is not None and eval_stats["glossary"]["total_tokens"]
        else None
    )

    rows = []
    for method in ("baseline", "glossary", "placeholder"):
        trans = translation[method]
        row = {
            "method": method,
            "translation_documents": trans["documents"],
            "translation_paragraphs": trans["paragraphs"],
            "translation_latency_seconds_sum": trans["latency_seconds_sum"],
            "translation_latency_seconds_mean": trans["latency_seconds_mean"],
            "translation_attempts_mean": trans["attempts_mean"],
            "generation_retries": trans["generation_retries"],
            "glossary_compliance_retries": trans["glossary_compliance_retries"],
            "semantic_verifier_calls": trans["semantic_verifier_calls"],
            "fallback_paragraphs": trans["fallback_paragraphs"],
            "unresolved_paragraphs": trans["unresolved_paragraphs"],
            "strategy_counts": str(trans["strategy_counts"]),
        }
        if method in eval_stats:
            ev = eval_stats[method]
            actual = costs.get(f"{method}_judge_actual")
            estimated = float(actual) if actual is not None else (ev["total_tokens"] * observed_rate if observed_rate is not None else None)
            row.update({
                "judge_documents": ev["documents"],
                "judge_paragraphs_successful": ev["paragraphs_successful"],
                "judge_paragraphs_failed": ev["paragraphs_failed"],
                "judge_prompt_tokens": ev["prompt_tokens"],
                "judge_completion_tokens": ev["completion_tokens"],
                "judge_total_tokens": ev["total_tokens"],
                "judge_tokens_per_successful_paragraph": ev["total_tokens"] / ev["paragraphs_successful"] if ev["paragraphs_successful"] else None,
                "judge_latency_seconds_sum": ev["judge_latency_seconds_sum"],
                "judge_latency_seconds_mean": ev["judge_latency_seconds_mean"],
                "judge_attempts_mean": ev["judge_attempts_mean"],
                "judge_actual_or_estimated_cost_cny": estimated,
                "judge_cost_is_actual": actual is not None,
                "judge_cost_per_1000_paragraphs_cny": estimated * 1000 / ev["paragraphs_successful"] if estimated is not None and ev["paragraphs_successful"] else None,
            })
        rows.append(row)

    write_csv(output_root / "10_efficiency_cost_table.csv", rows)
    write_json(output_root / "10_efficiency_cost_table.json", {
        "observed_judge_rate_cny_per_token": observed_rate,
        "rows": rows,
    })
    print(f"Observed judge rate: {observed_rate:.10f} CNY/token" if observed_rate is not None else "Observed judge rate: unavailable")
    print(f"Saved: {output_root / '10_efficiency_cost_table.csv'}")


if __name__ == "__main__":
    main()
