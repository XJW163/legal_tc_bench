from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from common import (
    benchmark_terms,
    index_documents,
    load_config,
    load_json,
    resolve,
    stable_order_key,
    terms_by_paragraph,
    translations_by_paragraph,
    write_csv,
    write_json,
)


def filter_term_list(terms: list[Any], selected_pids: set[str]) -> list[dict[str, Any]]:
    result = []
    for raw in terms:
        if not isinstance(raw, dict):
            continue
        item = copy.deepcopy(raw)
        occurrences = [
            occurrence
            for occurrence in (item.get("occurrences") or [])
            if isinstance(occurrence, dict)
            and str(occurrence.get("paragraph_id")) in selected_pids
        ]
        if occurrences:
            item["occurrences"] = occurrences
            result.append(item)
    return result


def subset_annotated(document: dict[str, Any], selected_pids: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["paragraphs"] = [
        item for item in (document.get("paragraphs") or [])
        if isinstance(item, dict) and str(item.get("paragraph_id")) in selected_pids
    ]
    if isinstance(document.get("benchmark_terms"), list):
        result["benchmark_terms"] = filter_term_list(document["benchmark_terms"], selected_pids)
    if isinstance(document.get("terms"), list):
        result["terms"] = filter_term_list(document["terms"], selected_pids)
    return result


def subset_translations(document: dict[str, Any], selected_pids: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["translations"] = [
        item for item in (document.get("translations") or [])
        if isinstance(item, dict) and str(item.get("paragraph_id")) in selected_pids
    ]
    result.pop("errors", None)
    result["subset_note"] = "Prepared for direct glossary-versus-placeholder blind evaluation."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact fixed sample for direct glossary-vs-placeholder evaluation.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--term-sample-size", type=int)
    parser.add_argument("--non-term-sample-size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths = config["paths"]
    settings = config.get("direct_ab") or {}
    term_n = args.term_sample_size if args.term_sample_size is not None else int(settings.get("term_sample_size", 500))
    non_term_n = args.non_term_sample_size if args.non_term_sample_size is not None else int(settings.get("non_term_sample_size", 200))
    seed = args.seed if args.seed is not None else int(settings.get("seed", 2026))

    annotated_index = index_documents(resolve(root, paths["annotated"]), "paragraphs")
    glossary_index = index_documents(resolve(root, paths["glossary"]), "translations")
    placeholder_index = index_documents(resolve(root, paths["placeholder"]), "translations")
    memory_index = index_documents(resolve(root, paths["term_memory"]), "terms")

    candidates: list[dict[str, Any]] = []
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for document_id, annotated_path in sorted(annotated_index.items()):
        if document_id not in glossary_index or document_id not in placeholder_index:
            continue
        annotated = load_json(annotated_path)
        glossary = load_json(glossary_index[document_id])
        placeholder = load_json(placeholder_index[document_id])
        loaded[document_id] = (annotated, glossary, placeholder)
        term_by_pid = terms_by_paragraph(annotated)
        g_by_pid = translations_by_paragraph(glossary)
        p_by_pid = translations_by_paragraph(placeholder)
        for paragraph in annotated.get("paragraphs") or []:
            if not isinstance(paragraph, dict) or "paragraph_id" not in paragraph:
                continue
            pid = str(paragraph.get("paragraph_id"))
            g_text = str((g_by_pid.get(pid) or {}).get("translation", "")).strip()
            p_text = str((p_by_pid.get(pid) or {}).get("translation", "")).strip()
            if not g_text or not p_text:
                continue
            candidates.append({
                "document_id": document_id,
                "paragraph_id": pid,
                "has_benchmark_terms": bool(term_by_pid.get(pid)),
                "source": str(paragraph.get("text", "")),
                "relevant_term_ids": " | ".join(sorted(str(x.get("term_id")) for x in term_by_pid.get(pid, []))),
                "relevant_source_terms": " | ".join(sorted(str(x.get("source_term")) for x in term_by_pid.get(pid, []))),
            })

    term_candidates = sorted(
        [row for row in candidates if row["has_benchmark_terms"]],
        key=lambda row: stable_order_key(seed, row["document_id"], row["paragraph_id"]),
    )
    non_term_candidates = sorted(
        [row for row in candidates if not row["has_benchmark_terms"]],
        key=lambda row: stable_order_key(seed, row["document_id"], row["paragraph_id"]),
    )
    if len(term_candidates) < term_n:
        raise ValueError(f"requested {term_n} term-bearing paragraphs, only {len(term_candidates)} available")
    if len(non_term_candidates) < non_term_n:
        raise ValueError(f"requested {non_term_n} non-term paragraphs, only {len(non_term_candidates)} available")

    selected = term_candidates[:term_n] + non_term_candidates[:non_term_n]
    selected.sort(key=lambda row: (row["document_id"], str(row["paragraph_id"])))
    selected_by_doc: dict[str, set[str]] = {}
    for row in selected:
        selected_by_doc.setdefault(row["document_id"], set()).add(str(row["paragraph_id"]))

    output_root = resolve(root, paths["output_root"]) / "direct_ab_sample"
    annotated_out = output_root / "annotated"
    glossary_out = output_root / "glossary"
    placeholder_out = output_root / "placeholder"
    memory_out = output_root / "term_memory"
    for directory in (annotated_out, glossary_out, placeholder_out, memory_out):
        directory.mkdir(parents=True, exist_ok=True)

    for document_id, pids in sorted(selected_by_doc.items()):
        annotated, glossary, placeholder = loaded[document_id]
        write_json(annotated_out / f"{document_id}.json", subset_annotated(annotated, pids))
        write_json(glossary_out / f"{document_id}.json", subset_translations(glossary, pids))
        write_json(placeholder_out / f"{document_id}.json", subset_translations(placeholder, pids))
        memory_path = memory_index.get(document_id)
        if memory_path is None:
            raise ValueError(f"missing term memory for selected document {document_id}")
        write_json(memory_out / f"{document_id}.json", load_json(memory_path))

    manifest = {
        "seed": seed,
        "requested_term_bearing": term_n,
        "requested_non_term": non_term_n,
        "selected_total": len(selected),
        "selected_term_bearing": sum(int(row["has_benchmark_terms"]) for row in selected),
        "selected_non_term": sum(int(not row["has_benchmark_terms"]) for row in selected),
        "documents": len(selected_by_doc),
        "paths": {
            "annotated": str(annotated_out),
            "glossary": str(glossary_out),
            "placeholder": str(placeholder_out),
            "term_memory": str(memory_out),
        },
    }
    write_json(output_root / "sample_manifest.json", manifest)
    write_csv(output_root / "sample_paragraphs.csv", selected)
    print(f"Selected {len(selected)} paragraphs across {len(selected_by_doc)} documents")
    print(f"  term-bearing: {manifest['selected_term_bearing']}")
    print(f"  non-term:     {manifest['selected_non_term']}")
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
