from __future__ import annotations

import argparse
from pathlib import Path

from common import atomic_json, check_legal_tc, experiment_registry, load_config, load_json, p, run, safe_ratio, write_csv


def controlled_manifest_row(item: dict, directory: Path) -> dict:
    manifest = load_json(directory / "controlled_translation_manifest.json", {}) or {}
    totals = manifest.get("totals") or {}
    total = int(totals.get("paragraphs_total", 0) or 0)
    uncontrolled = int(totals.get("uncontrolled_paragraphs", 0) or 0)
    controlled = max(0, total - uncontrolled)
    exact = int(totals.get("placeholder_exact_paragraphs", 0) or 0)
    retry = int(totals.get("placeholder_retry_success_paragraphs", 0) or 0)
    fallback = int(totals.get("placeholder_fallback_paragraphs", 0) or 0)
    unresolved = int(totals.get("unresolved_paragraphs", totals.get("placeholder_fallback_noncompliance_paragraphs", 0)) or 0)
    return {
        "experiment_id": item["experiment_id"],
        "model_key": item["model_key"],
        "method": item["method"],
        "manifest_status": manifest.get("status"),
        "documents_total": totals.get("documents_total"),
        "documents_complete": totals.get("documents_complete"),
        "paragraphs_total": total,
        "paragraph_errors": totals.get("paragraph_errors"),
        "controlled_paragraphs": controlled,
        "first_pass_rate": safe_ratio(exact, controlled),
        "placeholder_path_success_rate": safe_ratio(exact + retry, controlled),
        "fallback_rate": safe_ratio(fallback, controlled),
        "unresolved_rate": safe_ratio(unresolved, controlled),
        "fallback_noncompliance": totals.get("placeholder_fallback_noncompliance_paragraphs"),
        "glossary_noncompliance": totals.get("glossary_noncompliance_paragraphs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    annotated = p(root, config["paths"]["annotated"])
    results_root = p(root, config["paths"]["results_root"])
    validation_root = results_root / "validation"
    rows = []

    for item in experiment_registry(config, root, include_missing=True):
        directory = Path(item["translation_dir"])
        if not directory.is_dir():
            if args.skip_missing:
                rows.append({"experiment_id": item["experiment_id"], "status": "missing", "translation_dir": str(directory)})
                continue
            raise FileNotFoundError(directory)
        output = validation_root / f"{item['experiment_id']}.json"
        run([legal_tc, "validate-translations", annotated, directory, "--output", output], cwd=root)
        report = load_json(output, {}) or {}
        report_totals = report.get("totals") or {}
        row = {
            "experiment_id": item["experiment_id"],
            "model_key": item["model_key"],
            "method": item["method"],
            "translation_dir": str(directory),
            "validation_status": report.get("status"),
            "quality_status": report.get("quality_status"),
            "missing_documents": len(report.get("missing_documents") or []),
            "extra_documents": len(report.get("extra_documents") or []),
            "missing_paragraphs": report_totals.get("missing_paragraphs"),
            "empty_translations": report_totals.get("empty_translations"),
            "source_mismatches": report_totals.get("source_mismatches"),
            "paragraphs_with_latin_letters": report_totals.get("paragraphs_with_latin_letters"),
        }
        if item["method"] in {"glossary", "placeholder"}:
            row.update(controlled_manifest_row(item, directory))
        rows.append(row)

    write_csv(results_root / "experiment_validation.csv", rows)
    atomic_json(results_root / "experiment_validation.json", {"experiments": rows})
    print(f"wrote {results_root / 'experiment_validation.csv'}")


if __name__ == "__main__":
    main()
