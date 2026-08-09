from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, load_evaluation_rows, resolve, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the completed glossary and placeholder LLM-judge experiments.")
    parser.add_argument("--config", default="analysis_config.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    output_root = resolve(root, paths["output_root"])
    glossary_dir = resolve(root, paths["glossary_evaluation"])
    placeholder_dir = resolve(root, paths["placeholder_evaluation"])

    glossary, glossary_errors, glossary_docs = load_evaluation_rows(glossary_dir)
    placeholder, placeholder_errors, placeholder_docs = load_evaluation_rows(placeholder_dir)
    common = sorted(set(glossary) & set(placeholder))
    only_glossary = sorted(set(glossary) - set(placeholder))
    only_placeholder = sorted(set(placeholder) - set(glossary))

    mismatches = []
    for key in common:
        g = glossary[key]
        p = placeholder[key]
        checks = {
            "source": str(g.get("source", "")) == str(p.get("source", "")),
            "baseline_translation": str(g.get("baseline_translation", "")) == str(p.get("baseline_translation", "")),
            "has_benchmark_terms": bool(g.get("has_benchmark_terms")) == bool(p.get("has_benchmark_terms")),
        }
        for field, ok in checks.items():
            if not ok:
                mismatches.append({
                    "document_id": key[0],
                    "paragraph_id": key[1],
                    "field": field,
                    "glossary_value": g.get(field),
                    "placeholder_value": p.get(field),
                })

    failure_rows = []
    for method, errors in (("glossary", glossary_errors), ("placeholder", placeholder_errors)):
        for error in errors:
            failure_rows.append({
                "method": method,
                "document_id": error.get("document_id"),
                "paragraph_id": error.get("paragraph_id"),
                "error_type": error.get("error_type"),
                "error": error.get("error"),
                "prompt_tokens": (error.get("usage") or {}).get("prompt_tokens", 0),
                "completion_tokens": (error.get("usage") or {}).get("completion_tokens", 0),
                "total_tokens": (error.get("usage") or {}).get("total_tokens", 0),
                "updated_at": error.get("updated_at"),
            })

    audit = {
        "glossary": {
            "directory": str(glossary_dir),
            "documents": len(glossary_docs),
            "successful_paragraphs": len(glossary),
            "failed_paragraphs": len(glossary_errors),
        },
        "placeholder": {
            "directory": str(placeholder_dir),
            "documents": len(placeholder_docs),
            "successful_paragraphs": len(placeholder),
            "failed_paragraphs": len(placeholder_errors),
        },
        "paired": {
            "common_successful_paragraphs": len(common),
            "only_glossary_successful": len(only_glossary),
            "only_placeholder_successful": len(only_placeholder),
            "content_mismatches": len(mismatches),
        },
        "status": "ok" if not mismatches else "content_mismatch",
    }

    write_json(output_root / "00_current_evaluation_audit.json", audit)
    write_csv(output_root / "00_failed_paragraphs.csv", failure_rows)
    write_csv(output_root / "00_content_mismatches.csv", mismatches)
    write_csv(output_root / "00_only_glossary_successful.csv", [
        {"document_id": doc, "paragraph_id": pid} for doc, pid in only_glossary
    ])
    write_csv(output_root / "00_only_placeholder_successful.csv", [
        {"document_id": doc, "paragraph_id": pid} for doc, pid in only_placeholder
    ])

    print(f"Glossary:    {len(glossary)} successful, {len(glossary_errors)} failed, {len(glossary_docs)} documents")
    print(f"Placeholder: {len(placeholder)} successful, {len(placeholder_errors)} failed, {len(placeholder_docs)} documents")
    print(f"Common successful paragraphs: {len(common)}")
    print(f"Content mismatches: {len(mismatches)}")
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
