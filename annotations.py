from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TERM_TYPES = {
    "PARTY_NAME",
    "DEFINED_LEGAL_TERM",
    "RIGHT_OR_OBLIGATION",
    "LEGAL_CONCEPT",
    "COMMERCIAL_TERM",
    "DOCUMENT_REFERENCE",
    "GENERAL_TERM",
}
CRITICALITIES = {"CRITICAL", "IMPORTANT", "AUXILIARY", "EXCLUDED"}
TRUE_VALUES = {"1", "true", "yes", "y", "include"}
FALSE_VALUES = {"0", "false", "no", "n", "exclude", ""}

REVIEW_FIELDS = [
    "document_id",
    "term_id",
    "source_term",
    "frequency",
    "extraction_method",
    "definition_paragraph_ids",
    "definition_context",
    "suggested_term_type",
    "suggested_criticality",
    "suggested_include",
    "term_type",
    "criticality",
    "include_in_benchmark",
    "reviewer_notes",
]


def export_term_review(input_path: Path, output_csv: Path) -> int:
    """Export candidate terms from one dataset JSON or a directory to a review CSV."""
    rows: list[dict[str, str]] = []
    for path, doc in _iter_dataset_documents(input_path):
        document_id = str(doc.get("document_id") or path.stem)
        paragraphs = {
            int(p["paragraph_id"]): str(p.get("text", ""))
            for p in doc.get("paragraphs", [])
            if "paragraph_id" in p
        }
        for term in doc.get("terms", []):
            source_term = str(term.get("source_term", "")).strip()
            definition_ids = [int(x) for x in term.get("definition_paragraph_ids", [])]
            contexts = [paragraphs.get(pid, "") for pid in definition_ids]
            context = " || ".join(_shorten(x, 500) for x in contexts if x)
            suggested_type, suggested_criticality, suggested_include = suggest_annotation(source_term, context)
            annotation = term.get("annotation") or {}
            rows.append({
                "document_id": document_id,
                "term_id": str(term.get("term_id", "")),
                "source_term": source_term,
                "frequency": str(len(term.get("occurrences", []))),
                "extraction_method": str(term.get("extraction_method", "")),
                "definition_paragraph_ids": ";".join(map(str, definition_ids)),
                "definition_context": context,
                "suggested_term_type": suggested_type,
                "suggested_criticality": suggested_criticality,
                "suggested_include": "true" if suggested_include else "false",
                "term_type": str(annotation.get("term_type", "")),
                "criticality": str(annotation.get("criticality", "")),
                "include_in_benchmark": _format_optional_bool(annotation.get("include_in_benchmark")),
                "reviewer_notes": str(annotation.get("reviewer_notes", "")),
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def apply_term_review(
    input_path: Path,
    review_csv: Path,
    output_path: Path,
    *,
    accept_suggestions: bool = False,
    strict: bool = True,
) -> dict:
    """Apply reviewed CSV annotations and emit annotated JSON document(s)."""
    review = _load_review(review_csv, accept_suggestions=accept_suggestions)
    documents = list(_iter_dataset_documents(input_path))
    is_multi = input_path.is_dir() or len(documents) > 1
    if is_multi:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    totals = Counter()
    for source_path, doc in documents:
        document_id = str(doc.get("document_id") or source_path.stem)
        seen_keys: set[tuple[str, str]] = set()
        benchmark_terms: list[dict] = []
        for term in doc.get("terms", []):
            key = (document_id, str(term.get("term_id", "")))
            row = review.get(key)
            if row is None:
                totals["unreviewed"] += 1
                if strict:
                    raise ValueError(f"Missing review row for {document_id}/{key[1]}")
                continue
            seen_keys.add(key)
            if _normalise_term(term.get("source_term", "")) != _normalise_term(row["source_term"]):
                raise ValueError(
                    f"Source term mismatch for {document_id}/{key[1]}: "
                    f"dataset={term.get('source_term')!r}, CSV={row['source_term']!r}"
                )
            annotation = {
                "term_type": row["term_type"],
                "criticality": row["criticality"],
                "include_in_benchmark": row["include_in_benchmark"],
                "reviewer_notes": row["reviewer_notes"],
                "annotation_source": "human-review-csv" if not row["used_suggestion"] else "accepted-heuristic-suggestion",
            }
            term["annotation"] = annotation
            totals["reviewed"] += 1
            totals[f"type_{row['term_type']}"] += 1
            totals[f"criticality_{row['criticality']}"] += 1
            if row["include_in_benchmark"] and row["criticality"] != "EXCLUDED":
                benchmark_terms.append(term)
                totals["included"] += 1
            else:
                totals["excluded"] += 1

        extra = set(review) - seen_keys
        # Only report extras belonging to this document. They frequently indicate stale CSVs.
        extra_for_doc = sorted(k for k in extra if k[0] == document_id)
        if extra_for_doc and strict:
            raise ValueError(f"Review CSV contains unknown rows for {document_id}: {extra_for_doc[:5]}")

        doc["schema_version"] = "1.1"
        doc["benchmark_terms"] = benchmark_terms
        doc["annotation_statistics"] = _document_statistics(doc.get("terms", []), benchmark_terms)
        destination = output_path / source_path.name if is_multi else output_path
        destination.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({
            "document_id": document_id,
            "source": str(source_path),
            "output": str(destination),
            **doc["annotation_statistics"],
        })

    result = {"documents": manifest, "totals": dict(totals)}
    if is_multi:
        (output_path / "annotation_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def suggest_annotation(term: str, context: str = "") -> tuple[str, str, bool]:
    """Conservative heuristic pre-labels; human review remains authoritative."""
    value = re.sub(r"\s+", " ", term).strip()
    lower = value.casefold()

    company_markers = re.compile(
        r"\b(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|llc|plc|company|co\.?)\b",
        re.I,
    )
    party_words = {"buyer", "seller", "purchaser", "vendor", "landlord", "tenant", "licensor", "licensee", "employer", "employee"}
    document_words = ("agreement", "schedule", "exhibit", "appendix", "annex", "statement", "certificate", "notice")
    critical_patterns = (
        "effective date", "closing date", "completion date", "termination date",
        "governing law", "confidential information", "intellectual property",
        "indemnity", "indemnification", "liability", "warranty", "representation",
        "force majeure", "breach", "default", "consideration", "term",
    )
    right_patterns = ("right", "rights", "obligation", "obligations", "license", "licenses", "duty", "duties")
    commercial_patterns = ("price", "payment", "fee", "fees", "royalty", "deposit", "purchase", "product", "services")

    if company_markers.search(value) or (value.isupper() and 2 <= len(value) <= 8):
        return "PARTY_NAME", "IMPORTANT", False
    if lower in party_words:
        return "PARTY_NAME", "CRITICAL", True
    if any(x in lower for x in critical_patterns):
        return "DEFINED_LEGAL_TERM", "CRITICAL", True
    if any(x in lower for x in right_patterns):
        return "RIGHT_OR_OBLIGATION", "CRITICAL", True
    if any(x in lower for x in document_words):
        return "DOCUMENT_REFERENCE", "IMPORTANT", True
    if any(x in lower for x in commercial_patterns):
        return "COMMERCIAL_TERM", "IMPORTANT", True
    # Parenthetically defined capitalised terms are useful candidates but not automatically critical.
    if value[:1].isupper():
        return "DEFINED_LEGAL_TERM", "IMPORTANT", True
    return "GENERAL_TERM", "AUXILIARY", False


def _load_review(path: Path, *, accept_suggestions: bool) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [x for x in REVIEW_FIELDS if x not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Review CSV is missing columns: {missing}")
        for line_number, raw in enumerate(reader, 2):
            document_id = raw["document_id"].strip()
            term_id = raw["term_id"].strip()
            key = (document_id, term_id)
            if key in rows:
                raise ValueError(f"Duplicate review row at line {line_number}: {document_id}/{term_id}")

            term_type = raw["term_type"].strip().upper()
            criticality = raw["criticality"].strip().upper()
            include_text = raw["include_in_benchmark"].strip()
            used_suggestion = False
            if accept_suggestions:
                if not term_type:
                    term_type = raw["suggested_term_type"].strip().upper()
                    used_suggestion = True
                if not criticality:
                    criticality = raw["suggested_criticality"].strip().upper()
                    used_suggestion = True
                if not include_text:
                    include_text = raw["suggested_include"].strip()
                    used_suggestion = True

            if term_type not in TERM_TYPES:
                raise ValueError(f"Invalid term_type at line {line_number}: {term_type!r}")
            if criticality not in CRITICALITIES:
                raise ValueError(f"Invalid criticality at line {line_number}: {criticality!r}")
            include = _parse_bool(include_text, line_number)
            if criticality == "EXCLUDED":
                include = False

            rows[key] = {
                "source_term": raw["source_term"].strip(),
                "term_type": term_type,
                "criticality": criticality,
                "include_in_benchmark": include,
                "reviewer_notes": raw["reviewer_notes"].strip(),
                "used_suggestion": used_suggestion,
            }
    return rows


def _iter_dataset_documents(input_path: Path) -> Iterable[tuple[Path, dict]]:
    if input_path.is_file():
        yield input_path, _read_dataset(input_path)
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    for path in sorted(input_path.rglob("*.json")):
        if path.name.endswith("manifest.json"):
            continue
        try:
            doc = _read_dataset(path)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict) and "terms" in doc:
            yield path, doc


def _read_dataset(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "terms" not in value:
        raise ValueError(f"Not a LegalTC dataset document: {path}")
    return value


def _document_statistics(terms: list[dict], benchmark_terms: list[dict]) -> dict:
    type_counts = Counter()
    criticality_counts = Counter()
    reviewed = 0
    for term in terms:
        annotation = term.get("annotation")
        if not annotation:
            continue
        reviewed += 1
        type_counts[annotation["term_type"]] += 1
        criticality_counts[annotation["criticality"]] += 1
    return {
        "candidate_term_count": len(terms),
        "reviewed_term_count": reviewed,
        "benchmark_term_count": len(benchmark_terms),
        "term_type_counts": dict(type_counts),
        "criticality_counts": dict(criticality_counts),
    }


def _parse_bool(value: str, line_number: int) -> bool:
    normalized = value.strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid include_in_benchmark at line {line_number}: {value!r}")


def _format_optional_bool(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _normalise_term(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _shorten(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"
