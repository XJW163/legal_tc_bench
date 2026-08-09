from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_documents(root: Path, required_key: str) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
    result: dict[str, Path] = {}
    for path in candidates:
        if path.name.endswith(("manifest.json", ".audit.json", ".checkpoint.json", "summary.json")):
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or required_key not in value:
            continue
        document_id = str(value.get("document_id") or path.stem)
        if document_id in result:
            raise ValueError(f"duplicate document_id {document_id!r}: {result[document_id]} and {path}")
        result[document_id] = path
    return result


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = load_json(config_path)
    root_value = config.get("project_root", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return config, root


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def stable_fraction(seed: int, document_id: str, paragraph_id: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{document_id}\0{paragraph_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def stable_order_key(seed: int, document_id: str, paragraph_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{document_id}\0{paragraph_id}".encode("utf-8")).digest()


def benchmark_terms(document: dict[str, Any]) -> list[dict[str, Any]]:
    terms = document.get("benchmark_terms")
    if isinstance(terms, list):
        return [x for x in terms if isinstance(x, dict)]
    return [
        term
        for term in (document.get("terms") or [])
        if isinstance(term, dict)
        and (term.get("annotation") or {}).get("include_in_benchmark", False)
        and (term.get("annotation") or {}).get("criticality") != "EXCLUDED"
    ]


def terms_by_paragraph(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_pid: dict[str, dict[str, dict[str, Any]]] = {}
    for term in benchmark_terms(document):
        term_id = str(term.get("term_id", ""))
        item = {
            "term_id": term_id,
            "source_term": str(term.get("source_term", "")),
            "annotation": term.get("annotation") or {},
        }
        for occurrence in term.get("occurrences") or []:
            if not isinstance(occurrence, dict) or "paragraph_id" not in occurrence:
                continue
            pid = str(occurrence.get("paragraph_id"))
            by_pid.setdefault(pid, {})[term_id] = item
    return {pid: list(values.values()) for pid, values in by_pid.items()}


def translations_by_paragraph(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("paragraph_id")): item
        for item in (document.get("translations") or [])
        if isinstance(item, dict) and "paragraph_id" in item
    }


def iter_evaluation_documents(directory: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(directory.rglob("*.translation-evaluation.json")):
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            yield path, value


def load_evaluation_rows(directory: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    for path, data in iter_evaluation_documents(directory):
        document_id = str(data.get("document_id") or path.stem)
        documents[document_id] = data
        for row in data.get("evaluations") or []:
            if not isinstance(row, dict) or "paragraph_id" not in row:
                continue
            key = (document_id, str(row.get("paragraph_id")))
            if key in rows:
                raise ValueError(f"duplicate evaluation row: {key}")
            rows[key] = row
        for error in data.get("errors") or []:
            if isinstance(error, dict):
                errors.append({"document_id": document_id, **error})
    return rows, errors, documents


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None
