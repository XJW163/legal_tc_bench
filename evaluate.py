from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def locate_term_translations(
    dataset_path: Path,
    translation_path: Path,
    output_path: Path,
) -> int:
    """Pair each benchmark-term occurrence with its translated paragraph.

    This stage deliberately does not guess word alignment. It creates the
    occurrence-level input consumed by ``llm-extract-renderings``.
    """
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    translated = json.loads(translation_path.read_text(encoding="utf-8"))

    document_id = str(dataset.get("document_id") or dataset_path.stem)
    translated_document_id = str(translated.get("document_id") or "")
    if translated_document_id and translated_document_id != document_id:
        raise ValueError(
            f"document_id mismatch: dataset={document_id!r}, "
            f"translation={translated_document_id!r}"
        )

    by_pid = {
        int(item["paragraph_id"]): item
        for item in translated.get("translations", [])
        if "paragraph_id" in item
    }

    terms = dataset.get("benchmark_terms")
    if terms is None:
        terms = [
            term
            for term in dataset.get("terms", [])
            if (term.get("annotation") or {}).get("include_in_benchmark", False)
            and (term.get("annotation") or {}).get("criticality") != "EXCLUDED"
        ]

    rows: list[dict[str, Any]] = []
    for term in terms:
        annotation = term.get("annotation") or {}
        for occurrence_index, occurrence in enumerate(term.get("occurrences", [])):
            pid = int(occurrence["paragraph_id"])
            translation_item = by_pid.get(pid, {})
            target = str(translation_item.get("translation", ""))
            occurrence_id = (
                f"{document_id}::{term.get('term_id')}::{occurrence_index}::"
                f"{pid}::{occurrence.get('char_start', '')}"
            )
            rows.append({
                "occurrence_id": occurrence_id,
                "document_id": document_id,
                "term_id": str(term.get("term_id", "")),
                "source_term": str(term.get("source_term", "")),
                "term_type": annotation.get("term_type"),
                "criticality": annotation.get("criticality"),
                "paragraph_id": pid,
                "occurrence_index": occurrence_index,
                "char_start": occurrence.get("char_start"),
                "char_end": occurrence.get("char_end"),
                "source_context": occurrence.get("context", ""),
                "target_paragraph": target,
                "rendering": None,
                "extraction_status": None,
                "extraction_confidence": None,
                "extraction_rationale": None,
                "extraction_model": None,
                "label": None,
                "notes": (
                    None
                    if target
                    else "No translated paragraph found for this paragraph_id."
                ),
            })

    _atomic_json(output_path, rows)
    return len(rows)


def locate_all_term_translations(
    dataset_dir: Path,
    translation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create review JSON files for every matching dataset/translation pair."""
    datasets = _index_documents(dataset_dir, required_key="paragraphs")
    translations = _index_documents(translation_dir, required_key="translations")
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[dict[str, Any]] = []
    missing_translations: list[str] = []
    for document_id, dataset_path in sorted(datasets.items()):
        translation_path = translations.get(document_id)
        if translation_path is None:
            missing_translations.append(document_id)
            continue
        output_path = output_dir / f"{document_id}.review.json"
        row_count = locate_term_translations(
            dataset_path,
            translation_path,
            output_path,
        )
        created.append({
            "document_id": document_id,
            "dataset": str(dataset_path),
            "translation": str(translation_path),
            "output": str(output_path),
            "occurrence_count": row_count,
        })

    manifest = {
        "dataset_count": len(datasets),
        "translation_count": len(translations),
        "created_count": len(created),
        "occurrence_count": sum(x["occurrence_count"] for x in created),
        "missing_translation_count": len(missing_translations),
        "missing_translations": missing_translations,
        "documents": created,
    }
    _atomic_json(output_dir / "review_manifest.json", manifest)
    return manifest


def compute_metrics(review_path: Path, output_path: Path) -> None:
    rows = json.loads(review_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[str]] = {}
    source_terms: dict[tuple[str, str], str] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        rendering = normalize_rendering(row.get("rendering"))
        if not rendering:
            continue
        key = (str(row.get("document_id", "")), str(row["term_id"]))
        grouped.setdefault(key, []).append(rendering)
        source_terms[key] = row["source_term"]
        metadata[key] = {
            "term_type": row.get("term_type"),
            "criticality": row.get("criticality"),
        }

    term_metrics = []
    for key, renderings in grouped.items():
        document_id, term_id = key
        counts = Counter(renderings)
        n = len(renderings)
        k = len(counts)
        dominant = counts.most_common(1)[0][1]
        dtr = dominant / n
        tvr = (k - 1) / max(1, n - 1)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        normalized_entropy = entropy / math.log2(k) if k > 1 else 0.0
        term_metrics.append({
            "document_id": document_id,
            "term_id": term_id,
            "source_term": source_terms[key],
            **metadata[key],
            "occurrences_reviewed": n,
            "variant_count": k,
            "dominant_translation_ratio": round(dtr, 6),
            "translation_variation_rate": round(tvr, 6),
            "entropy_bits": round(entropy, 6),
            "normalized_entropy": round(normalized_entropy, 6),
            "renderings": dict(counts),
        })
    term_metrics.sort(
        key=lambda x: (
            x["dominant_translation_ratio"],
            -x["occurrences_reviewed"],
            x["document_id"],
            x["term_id"],
        )
    )
    _atomic_json(output_path, {"terms": term_metrics})


def normalize_rendering(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"[\s\u3000]+", "", value)
    value = value.strip("，。；：、,. ;:()（）[]【】\"'“”")
    return value or None


def _index_documents(directory: Path, *, required_key: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.json")):
        if path.name.endswith(("manifest.json", ".audit.json", ".checkpoint.json")):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or required_key not in value:
            continue
        document_id = str(value.get("document_id") or path.stem)
        if document_id in result:
            raise ValueError(
                f"Duplicate document_id {document_id!r}: "
                f"{result[document_id]} and {path}"
            )
        result[document_id] = path
    return result


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)
