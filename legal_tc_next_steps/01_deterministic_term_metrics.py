from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    benchmark_terms,
    index_documents,
    load_config,
    load_json,
    resolve,
    translations_by_paragraph,
    write_csv,
    write_json,
)

SPACE_RE = re.compile(r"[\s\u3000]+")


def norm(value: Any) -> str:
    return SPACE_RE.sub("", str(value or "")).strip()


def variant_renderings(term_memory_item: dict[str, Any], key: str) -> list[str]:
    values = []
    for item in term_memory_item.get(key) or []:
        if isinstance(item, dict):
            rendering = norm(item.get("rendering"))
        else:
            rendering = norm(item)
        if rendering:
            values.append(rendering)
    return values


def memory_by_term(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    data = load_json(path)
    document_id = str(data.get("document_id") or path.stem)
    return document_id, {
        str(item.get("term_id")): item
        for item in (data.get("terms") or [])
        if isinstance(item, dict) and item.get("term_id") is not None
    }


def safe_rate(num: int, den: int) -> float | None:
    return num / den if den else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deterministic string-presence terminology metrics.")
    parser.add_argument("--config", default="analysis_config.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    annotated_index = index_documents(resolve(root, paths["annotated"]), "paragraphs")
    memory_index = index_documents(resolve(root, paths["term_memory"]), "terms")
    method_paths = {
        "baseline": resolve(root, paths["baseline"]),
        "glossary": resolve(root, paths["glossary"]),
        "placeholder": resolve(root, paths["placeholder"]),
    }
    method_indexes = {name: index_documents(path, "translations") for name, path in method_paths.items()}
    output_root = resolve(root, paths["output_root"])

    occurrence_rows: list[dict[str, Any]] = []
    per_method_term: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        method: defaultdict(list) for method in method_indexes
    }
    per_method_doc: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: defaultdict(list) for method in method_indexes
    }
    per_method_paragraph: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        method: defaultdict(list) for method in method_indexes
    }

    for document_id, annotated_path in sorted(annotated_index.items()):
        memory_path = memory_index.get(document_id)
        if memory_path is None:
            raise ValueError(f"missing term memory for {document_id}")
        _, memory = memory_by_term(memory_path)
        annotated = load_json(annotated_path)
        translations = {}
        for method, index in method_indexes.items():
            path = index.get(document_id)
            if path is None:
                translations[method] = {}
            else:
                translations[method] = translations_by_paragraph(load_json(path))

        for term in benchmark_terms(annotated):
            term_id = str(term.get("term_id", ""))
            source_term = str(term.get("source_term", ""))
            mem = memory.get(term_id, {})
            canonical = norm(mem.get("canonical_translation"))
            allowed = variant_renderings(mem, "allowed_variants")
            risky = variant_renderings(mem, "risky_variants")
            acceptable = list(dict.fromkeys([x for x in [canonical, *allowed] if x]))

            for occurrence_index, occurrence in enumerate(term.get("occurrences") or []):
                if not isinstance(occurrence, dict) or "paragraph_id" not in occurrence:
                    continue
                paragraph_id = str(occurrence.get("paragraph_id"))
                for method, by_pid in translations.items():
                    translation = str((by_pid.get(paragraph_id) or {}).get("translation", ""))
                    target = norm(translation)
                    row = {
                        "method": method,
                        "document_id": document_id,
                        "term_id": term_id,
                        "source_term": source_term,
                        "paragraph_id": paragraph_id,
                        "occurrence_index": occurrence_index,
                        "canonical_translation": canonical,
                        "allowed_variants": " | ".join(allowed),
                        "risky_variants": " | ".join(risky),
                        "translation_present": bool(target),
                        "canonical_present": bool(canonical and canonical in target),
                        "acceptable_present": any(item in target for item in acceptable),
                        "risky_present": any(item in target for item in risky),
                        "target_translation": translation,
                    }
                    occurrence_rows.append(row)
                    per_method_term[method][(document_id, term_id)].append(row)
                    per_method_doc[method][document_id].append(row)
                    per_method_paragraph[method][(document_id, paragraph_id)].append(row)

    summary_rows = []
    detailed = {}
    for method in method_indexes:
        rows = [row for row in occurrence_rows if row["method"] == method]
        total = len(rows)
        translated = sum(int(row["translation_present"]) for row in rows)
        canonical = sum(int(row["canonical_present"]) for row in rows)
        acceptable = sum(int(row["acceptable_present"]) for row in rows)
        risky = sum(int(row["risky_present"]) for row in rows)

        term_groups = per_method_term[method]
        doc_groups = per_method_doc[method]
        paragraph_groups = per_method_paragraph[method]
        strict_canonical_terms = sum(all(x["canonical_present"] for x in group) for group in term_groups.values())
        strict_acceptable_terms = sum(all(x["acceptable_present"] for x in group) for group in term_groups.values())
        strict_canonical_docs = sum(all(x["canonical_present"] for x in group) for group in doc_groups.values())
        strict_acceptable_docs = sum(all(x["acceptable_present"] for x in group) for group in doc_groups.values())
        all_canonical_paragraphs = sum(all(x["canonical_present"] for x in group) for group in paragraph_groups.values())
        all_acceptable_paragraphs = sum(all(x["acceptable_present"] for x in group) for group in paragraph_groups.values())

        item = {
            "method": method,
            "occurrences": total,
            "translation_coverage": safe_rate(translated, total),
            "canonical_presence_micro": safe_rate(canonical, total),
            "acceptable_presence_micro": safe_rate(acceptable, total),
            "risky_variant_presence_micro": safe_rate(risky, total),
            "terms": len(term_groups),
            "strict_canonical_term_rate": safe_rate(strict_canonical_terms, len(term_groups)),
            "strict_acceptable_term_rate": safe_rate(strict_acceptable_terms, len(term_groups)),
            "documents": len(doc_groups),
            "strict_canonical_document_rate": safe_rate(strict_canonical_docs, len(doc_groups)),
            "strict_acceptable_document_rate": safe_rate(strict_acceptable_docs, len(doc_groups)),
            "term_bearing_paragraphs": len(paragraph_groups),
            "all_canonical_paragraph_rate": safe_rate(all_canonical_paragraphs, len(paragraph_groups)),
            "all_acceptable_paragraph_rate": safe_rate(all_acceptable_paragraphs, len(paragraph_groups)),
        }
        summary_rows.append(item)
        detailed[method] = item

    write_csv(output_root / "01_deterministic_term_metrics.csv", summary_rows)
    write_csv(output_root / "01_deterministic_term_occurrences.csv", occurrence_rows)
    write_json(output_root / "01_deterministic_term_metrics.json", {"methods": detailed})

    print("Deterministic string-presence metrics")
    for row in summary_rows:
        print(
            f"{row['method']:12s} canonical={row['canonical_presence_micro']:.4f} "
            f"acceptable={row['acceptable_presence_micro']:.4f} "
            f"strict_term={row['strict_acceptable_term_rate']:.4f} "
            f"strict_doc={row['strict_acceptable_document_rate']:.4f}"
        )
    print(f"Saved under: {output_root}")
    print("Note: these are exact string-presence metrics, not semantic alignment metrics.")


if __name__ == "__main__":
    main()
