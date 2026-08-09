from __future__ import annotations

import argparse
from pathlib import Path

from common import atomic_json, controlled_experiment_name, load_config, load_json, p, safe_ratio, write_csv


def sum_latency(experiment_dir: Path) -> float:
    total = 0.0
    for path in experiment_dir.glob("*.controlled.translation.json"):
        data = load_json(path, {}) or {}
        for row in data.get("translations", []):
            value = row.get("latency_seconds")
            if isinstance(value, (int, float)):
                total += float(value)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    controlled_root = p(root, config["paths"]["controlled_root"])
    results_root = p(root, config["paths"]["results_root"])
    rows = []
    for model in config["models"]:
        key = model["key"]
        for method in ("glossary", "placeholder"):
            experiment = controlled_experiment_name(key, method)
            directory = controlled_root / experiment
            manifest = load_json(directory / "controlled_translation_manifest.json", {}) or {}
            totals = manifest.get("totals") or {}
            total = int(totals.get("paragraphs_total", 0) or 0)
            uncontrolled = int(totals.get("uncontrolled_paragraphs", 0) or 0)
            controlled = max(0, total - uncontrolled)
            exact = int(totals.get("placeholder_exact_paragraphs", 0) or 0)
            retry = int(totals.get("placeholder_retry_success_paragraphs", 0) or 0)
            fallback = int(totals.get("placeholder_fallback_paragraphs", 0) or 0)
            noncompliant = int(totals.get("placeholder_fallback_noncompliance_paragraphs", 0) or 0)
            unresolved = int(totals.get("unresolved_paragraphs", noncompliant) or 0)
            latency = sum_latency(directory) if directory.is_dir() else 0.0
            rows.append({
                "model_key": key,
                "model": model["model"],
                "method": method,
                "experiment": experiment,
                "manifest_status": manifest.get("status", "missing"),
                "documents_total": totals.get("documents_total"),
                "documents_complete": totals.get("documents_complete"),
                "paragraphs_total": total,
                "controlled_paragraphs": controlled,
                "controlled_occurrences": totals.get("controlled_term_occurrences"),
                "first_pass_exact": exact,
                "retry_success": retry,
                "fallback": fallback,
                "unresolved": unresolved,
                "first_pass_rate": safe_ratio(exact, controlled),
                "retry_recovery_rate": safe_ratio(retry, int(totals.get("placeholder_retries", 0) or 0)),
                "placeholder_path_success_rate": safe_ratio(exact + retry, controlled),
                "fallback_rate": safe_ratio(fallback, controlled),
                "final_compliance_rate": safe_ratio(controlled - unresolved, controlled),
                "unresolved_rate": safe_ratio(unresolved, controlled),
                "quality_gate_rejections": totals.get("placeholder_quality_gate_rejections"),
                "inventory_rejections": totals.get("placeholder_inventory_rejections"),
                "context_rejections": totals.get("placeholder_context_rejections"),
                "restored_rejections": totals.get("placeholder_restored_rejections"),
                "semantic_rejections": totals.get("placeholder_semantic_rejections"),
                "verifier_calls": totals.get("placeholder_semantic_verifier_calls"),
                "generation_retries": totals.get("generation_retries"),
                "total_latency_seconds": round(latency, 3),
                "seconds_per_document": safe_ratio(latency, int(totals.get("documents_complete", 0) or 0)),
            })

    write_csv(results_root / "controlled_reliability.csv", rows)
    atomic_json(results_root / "controlled_reliability.json", {"experiments": rows})
    print(f"wrote {results_root / 'controlled_reliability.csv'}")


if __name__ == "__main__":
    main()
