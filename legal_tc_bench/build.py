from __future__ import annotations

import json
from pathlib import Path

from .terms import extract_defined_terms
from .text import parse_document


def build_document(raw_path: Path, metadata_path: Path, output_path: Path, min_occurrences: int = 3) -> dict:
    paragraphs = parse_document(raw_path)
    terms = extract_defined_terms(paragraphs, min_occurrences=min_occurrences)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    doc = {
        "schema_version": "1.0",
        "document_id": raw_path.parent.name + "_" + raw_path.stem,
        "source_file": str(raw_path),
        "metadata": metadata,
        "paragraphs": [{"paragraph_id": i, "text": p} for i, p in enumerate(paragraphs)],
        "terms": [t.to_dict() for t in terms],
        "statistics": {
            "paragraph_count": len(paragraphs),
            "term_count": len(terms),
            "term_occurrence_count": sum(len(t.occurrences) for t in terms),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc
