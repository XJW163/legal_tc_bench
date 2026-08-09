from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io_utils import atomic_write_json

VALIDATOR_VERSION = "definition-use-rules-v2.1"
CONTROL_LEVELS = {"HARD", "SOFT", "NONE"}
USAGE_STATUSES = {
    "DEFINED_TERM_USAGE",
    "DEFINIENDUM",
    "EXTERNAL_MODIFIED_DEFINED_TERM",
    "ENTITY_RELATION_USAGE",
    "COMMON_WORD_USAGE",
    "OTHER_LEGAL_NAME_COMPONENT",
    "OTHER_NAMED_EXPRESSION_COMPONENT",
    "POSSIBLY_NESTED_USAGE",
    "INVALID_SPAN",
    "SURFACE_MISMATCH",
}

# Terms such as Code are especially likely to occur as the head of a different
# statute or instrument name.  These are conservative exclusion patterns for the
# short-term occurrence only; the larger phrase remains ordinary source text.
_OTHER_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "code": (
        re.compile(r"\bInternal\s+Revenue\s+Code\b", re.I),
        re.compile(r"\bUnited\s+States?\s+Code\b", re.I),
        re.compile(r"\bU\.S\.\s+Code\b", re.I),
        re.compile(r"\bBankruptcy\s+Code\b", re.I),
        re.compile(r"\bCommercial\s+Code\b", re.I),
        re.compile(r"\bPenal\s+Code\b", re.I),
        re.compile(r"\bCivil\s+Code\b", re.I),
        re.compile(r"\bTax\s+Code\b", re.I),
    ),
    "act": (
        re.compile(r"\b[A-Z][A-Za-z.&'’\-]*(?:\s+[A-Z][A-Za-z.&'’\-]*){1,8}\s+Act\b"),
    ),
}

# A connector must be followed by another alphanumeric character. This prevents
# sentence-final punctuation from becoming part of an adjacent token; for example,
# ``Benefits. Gilead`` must not be classified as a compound expression.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.&'’\-][A-Za-z0-9]+)*")
_DEFINITION_VERB_RE = re.compile(r"\b(?:means|shall\s+mean|has\s+the\s+meaning)\b", re.I)
_QUOTE_CHARS = '"“”'

_DETERMINER_OR_POSSESSIVE_MODIFIERS = {
    "a", "an", "the", "any", "each", "every", "no", "such", "other",
    "this", "that", "these", "those", "either", "neither", "all", "some",
    "your", "his", "her", "its", "our", "their", "my",
}
_SAFE_EXTERNAL_MODIFIERS = {
    "annual", "existing", "current", "applicable", "respective", "aggregate",
    "total", "additional", "initial", "subsequent",
}
_ALL_CAPS_HEADING_TOKENS = {
    "policy", "policies", "benefits", "compensation", "bonus", "awards",
    "section", "article", "schedule", "exhibit", "appendix", "notice",
    "employment", "severance", "retirement", "confidentiality", "definitions",
}
_ENTITY_FULL_NAME_SUFFIXES = {
    "sciences", "properties", "corporation", "corp", "company", "inc", "incorporated",
    "limited", "ltd", "llc", "plc", "lp", "partnership", "holdings", "group", "bank",
    "bancorp", "trust", "association", "foundation", "university", "institute",
}
_ENTITY_RELATION_NOUNS = {
    "leadership", "team", "board", "employees", "employee", "management", "factory",
    "office", "facility", "facilities", "site", "sites", "policy", "policies", "plan",
    "plans", "benefits", "committee", "committees", "stock", "shares",
}
_GENERIC_NAMED_HEADS = {
    "plan", "policy", "program", "scheme", "bank", "committee", "fund", "award",
}
_LEGAL_INSTRUMENT_HEADS = {
    "act", "code", "regulation", "regulations", "rules", "statute", "order",
    "policy", "plan", "program", "agreement",
}
_JOB_TITLE_PATTERN = re.compile(
    r"\b(?:Chief|Senior|Executive|Vice|Deputy|Assistant|Managing|General|Regional|Group|Corporate)"
    r"(?:\s+(?:Chief|Senior|Executive|Vice|Deputy|Assistant|Managing|General|Regional|Group|Corporate))*"
    r"\s+(?:Officer|President|Chairman|Chairwoman|Director|Counsel|Secretary|Treasurer)\b"
)


def validate_term_occurrence_tree(
    annotated: Path,
    output: Path,
    *,
    overwrite: bool = False,
    max_documents: int | None = None,
    document_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate candidate defined-term occurrences before control injection.

    The validator is intentionally conservative.  It assigns one of three control
    levels to every candidate occurrence:

    * HARD: safe for occurrence-unique placeholder replacement;
    * SOFT: potentially a defined-term use, but only a conditional glossary hint;
    * NONE: common-word use, a component of another legal name, or invalid data.
    """
    annotated = Path(annotated)
    output = Path(output)
    documents = list(_iter_documents(annotated))
    if document_ids:
        documents = [x for x in documents if str(x[1].get("document_id") or x[0].stem) in document_ids]
    if max_documents is not None:
        if max_documents < 1:
            raise ValueError("max_documents must be >= 1")
        documents = documents[:max_documents]

    multi = annotated.is_dir() or len(documents) > 1
    if multi:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    aggregate_status_counts: Counter[str] = Counter()
    aggregate_reason_counts: Counter[str] = Counter()
    aggregate_context_counts: Counter[str] = Counter()
    for source_path, document in documents:
        document_id = str(document.get("document_id") or source_path.stem)
        destination = (
            output / f"{_safe_slug(document_id)}.occurrence-validation.json"
            if multi
            else output
        )
        if destination.exists() and not overwrite:
            existing = _load_json(destination)
            if existing and existing.get("source_signature") == _document_signature(document):
                result = existing
            else:
                raise ValueError(
                    f"occurrence validation already exists but source changed: {destination}; "
                    "use --overwrite or a new output"
                )
        else:
            result = validate_document_occurrences(document, source_path=source_path)
            atomic_write_json(destination, result)

        stats = result.get("statistics", {})
        for key, value in stats.items():
            if isinstance(value, int):
                totals[key] += value
        aggregate_status_counts.update(stats.get("usage_status_counts", {}))
        aggregate_reason_counts.update(stats.get("validation_reason_counts", {}))
        aggregate_context_counts.update(stats.get("context_classification_counts", {}))
        manifest_rows.append({
            "document_id": document_id,
            "source": str(source_path),
            "output": str(destination),
            "statistics": stats,
        })

    manifest = {
        "schema_version": "1.0",
        "validator_version": VALIDATOR_VERSION,
        "status": "complete",
        "annotated": str(annotated),
        "output": str(output),
        "documents_total": len(documents),
        "documents": manifest_rows,
        "totals": dict(totals),
        "aggregate_usage_status_counts": dict(aggregate_status_counts),
        "aggregate_validation_reason_counts": dict(aggregate_reason_counts),
        "aggregate_context_classification_counts": dict(aggregate_context_counts),
    }
    if multi:
        atomic_write_json(output / "occurrence_validation_manifest.json", manifest)
    return manifest


def validate_document_occurrences(
    document: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    document_id = str(document.get("document_id") or (source_path.stem if source_path else ""))
    paragraphs = {
        int(row.get("paragraph_id")): str(row.get("text", ""))
        for row in document.get("paragraphs", [])
        if _safe_int(row.get("paragraph_id")) is not None
    }

    terms_out: list[dict[str, Any]] = []
    level_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    total = 0

    for term in document.get("benchmark_terms") or []:
        source_term = str(term.get("source_term", "")).strip()
        term_id = str(term.get("term_id", ""))
        definition_pids = {
            int(pid) for pid in term.get("definition_paragraph_ids", [])
            if _safe_int(pid) is not None
        }
        occurrences_out: list[dict[str, Any]] = []
        for occurrence in term.get("occurrences", []):
            pid = _safe_int(occurrence.get("paragraph_id"))
            paragraph = paragraphs.get(pid, "") if pid is not None else ""
            validated = validate_occurrence(
                paragraph,
                source_term=source_term,
                term_id=term_id,
                occurrence=occurrence,
                is_definition_paragraph=pid in definition_pids if pid is not None else False,
            )
            occurrences_out.append(validated)
            total += 1
            level_counts[validated["control_eligibility"]] += 1
            status_counts[validated["usage_status"]] += 1
            for reason in validated.get("validation_reasons", []):
                reason_counts[reason] += 1
            context_classification = validated.get("context_classification")
            if context_classification:
                context_counts[str(context_classification)] += 1

        terms_out.append({
            "term_id": term_id,
            "source_term": source_term,
            "definition_paragraph_ids": sorted(definition_pids),
            "annotation": term.get("annotation", {}),
            "occurrences": occurrences_out,
        })

    return {
        "schema_version": "1.0",
        "validator_version": VALIDATOR_VERSION,
        "document_id": document_id,
        "source_path": str(source_path) if source_path else None,
        "source_signature": _document_signature(document),
        "terms": terms_out,
        "statistics": {
            "term_count": len(terms_out),
            "occurrence_count": total,
            "hard_occurrences": level_counts["HARD"],
            "soft_occurrences": level_counts["SOFT"],
            "excluded_occurrences": level_counts["NONE"],
            "usage_status_counts": dict(status_counts),
            "validation_reason_counts": dict(reason_counts),
            "context_classification_counts": dict(context_counts),
        },
    }


def validate_occurrence(
    paragraph: str,
    *,
    source_term: str,
    term_id: str,
    occurrence: dict[str, Any],
    is_definition_paragraph: bool,
) -> dict[str, Any]:
    result = dict(occurrence)
    result["term_id"] = term_id
    result["source_term"] = source_term

    start = _safe_int(occurrence.get("char_start"))
    end = _safe_int(occurrence.get("char_end"))
    if start is None or end is None or start < 0 or end <= start or end > len(paragraph):
        return _finish(
            result,
            surface="",
            usage_status="INVALID_SPAN",
            control="NONE",
            confidence=1.0,
            definition_role="NONE",
            reasons=["INVALID_SPAN"],
        )

    surface = paragraph[start:end]
    expected = _normalise_surface(source_term)
    actual = _normalise_surface(surface)
    if actual.casefold() != expected.casefold():
        return _finish(
            result,
            surface=surface,
            usage_status="SURFACE_MISMATCH",
            control="NONE",
            confidence=1.0,
            definition_role="NONE",
            reasons=["SOURCE_SLICE_DOES_NOT_MATCH_TERM"],
        )

    # Defined terms are case-significant in contracts.  A lowercase occurrence of a
    # capitalised defined term is normally an ordinary lexical use (Cause/cause).
    if actual != expected:
        return _finish(
            result,
            surface=surface,
            usage_status="COMMON_WORD_USAGE",
            control="NONE",
            confidence=0.99,
            definition_role="NONE",
            reasons=["CASE_MISMATCH_WITH_DEFINED_TERM"],
        )

    defining_spans = _definiendum_spans(paragraph, source_term)
    if any(start == a and end == b for a, b in defining_spans):
        return _finish(
            result,
            surface=surface,
            usage_status="DEFINIENDUM",
            control="HARD",
            confidence=1.0,
            definition_role="DEFINIENDUM",
            reasons=["EXPLICIT_DEFINIENDUM"],
        )

    other_name = _other_legal_name_span(paragraph, source_term, start, end)
    if other_name is not None:
        role = "DEFINIENS_COMPONENT" if is_definition_paragraph else "OTHER_NAME_COMPONENT"
        return _finish(
            result,
            surface=surface,
            usage_status="OTHER_LEGAL_NAME_COMPONENT",
            control="NONE",
            confidence=0.99,
            definition_role=role,
            reasons=["COMPONENT_OF_DIFFERENT_LEGAL_NAME"],
            compound_context=other_name,
        )

    compound_decision = _classify_compound_usage(paragraph, source_term, start, end)
    if compound_decision is not None:
        return _finish(
            result,
            surface=surface,
            usage_status=compound_decision["usage_status"],
            control=compound_decision["control"],
            confidence=compound_decision["confidence"],
            definition_role=compound_decision["definition_role"],
            reasons=compound_decision["reasons"],
            compound_context=compound_decision.get("compound_context"),
            context_classification=compound_decision.get("context_classification"),
        )

    return _finish(
        result,
        surface=surface,
        usage_status="DEFINED_TERM_USAGE",
        control="HARD",
        confidence=0.98,
        definition_role="OTHER",
        reasons=["EXACT_CASE_STANDALONE_MATCH"],
    )


def load_occurrence_validations(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.occurrence-validation.json"))
    values: dict[str, dict[str, Any]] = {}
    for file in files:
        data = _load_json(file)
        if not isinstance(data, dict):
            continue
        document_id = str(data.get("document_id") or "")
        if document_id:
            values[document_id] = data
    return values


def occurrence_map_by_paragraph(validation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for term in validation.get("terms", []):
        term_id = str(term.get("term_id", ""))
        source_term = str(term.get("source_term", ""))
        for occurrence in term.get("occurrences", []):
            item = dict(occurrence)
            item.setdefault("term_id", term_id)
            item.setdefault("source_term", source_term)
            result.setdefault(str(item.get("paragraph_id")), []).append(item)
    return result


def _definiendum_spans(paragraph: str, source_term: str) -> list[tuple[int, int]]:
    escaped = re.escape(source_term)
    patterns = [
        re.compile(rf"[{re.escape(_QUOTE_CHARS)}](?P<term>{escaped})[{re.escape(_QUOTE_CHARS)}]\s+(?:means|shall\s+mean|has\s+the\s+meaning)\b", re.I),
        re.compile(rf"\(\s*(?:the\s+)?[{re.escape(_QUOTE_CHARS)}](?P<term>{escaped})[{re.escape(_QUOTE_CHARS)}]\s*\)", re.I),
        re.compile(rf"(?<![A-Za-z0-9])(?P<term>{escaped})(?![A-Za-z0-9])\s+(?:means|shall\s+mean|has\s+the\s+meaning)\b"),
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(paragraph):
            spans.append(match.span("term"))
    return sorted(set(spans))


def _other_legal_name_span(paragraph: str, source_term: str, start: int, end: int) -> str | None:
    patterns = _OTHER_NAME_PATTERNS.get(source_term.casefold(), ())
    for pattern in patterns:
        for match in pattern.finditer(paragraph):
            if match.start() <= start and end <= match.end() and (match.start(), match.end()) != (start, end):
                return match.group(0)
    return None


def _classify_compound_usage(
    paragraph: str,
    source_term: str,
    start: int,
    end: int,
) -> dict[str, Any] | None:
    """Classify adjacent capitalised context with a high-precision policy.

    Components of a full title or legal instrument are excluded. External syntax
    such as possessives and determiners keeps the core term eligible for HARD
    control. Full entity names are excluded, while entity aliases used to modify an
    organisational relation noun remain HARD. Only unresolved compounds stay SOFT.
    """
    context = _adjacent_context(paragraph, start, end)
    left = context.get("left")
    right = context.get("right")
    surface = paragraph[start:end]

    title = _containing_job_title(paragraph, start, end)
    if title is not None:
        return {
            "usage_status": "OTHER_NAMED_EXPRESSION_COMPONENT",
            "control": "NONE",
            "confidence": 0.99,
            "definition_role": "OTHER_NAME_COMPONENT",
            "reasons": ["COMPONENT_OF_JOB_TITLE"],
            "compound_context": title,
            "context_classification": "JOB_TITLE",
        }

    legal_name = _containing_legal_instrument_name(paragraph, source_term, start, end, context)
    if legal_name is not None:
        return {
            "usage_status": "OTHER_NAMED_EXPRESSION_COMPONENT",
            "control": "NONE",
            "confidence": 0.99,
            "definition_role": "OTHER_NAME_COMPONENT",
            "reasons": ["COMPONENT_OF_LEGAL_INSTRUMENT_NAME"],
            "compound_context": legal_name,
            "context_classification": "LEGAL_INSTRUMENT_NAME",
        }

    # A possessive suffix belongs to the surrounding syntax, not to the defined
    # term itself: ``Gilead's Board`` may safely protect only the Gilead span.
    if _has_possessive_suffix(paragraph, end):
        return {
            "usage_status": "EXTERNAL_MODIFIED_DEFINED_TERM",
            "control": "HARD",
            "confidence": 0.99,
            "definition_role": "OTHER",
            "reasons": ["POSSESSIVE_SUFFIX_OUTSIDE_TERM"],
            "compound_context": _join_context(left, surface, None),
            "context_classification": "EXTERNAL_POSSESSIVE",
        }

    if left is not None and _is_possessive_token(left["token"]):
        return {
            "usage_status": "EXTERNAL_MODIFIED_DEFINED_TERM",
            "control": "HARD",
            "confidence": 0.99,
            "definition_role": "OTHER",
            "reasons": ["POSSESSIVE_MODIFIER_OUTSIDE_TERM"],
            "compound_context": _join_context(left, surface, None),
            "context_classification": "EXTERNAL_POSSESSIVE",
        }

    if left is not None and left["token"].casefold() in _DETERMINER_OR_POSSESSIVE_MODIFIERS:
        return {
            "usage_status": "EXTERNAL_MODIFIED_DEFINED_TERM",
            "control": "HARD",
            "confidence": 0.99,
            "definition_role": "OTHER",
            "reasons": ["DETERMINER_OR_POSSESSIVE_MODIFIER"],
            "compound_context": _join_context(left, surface, None),
            "context_classification": "EXTERNAL_DETERMINER",
        }

    # An all-caps section/heading label immediately before an entity alias is a
    # layout boundary, not a semantic modifier. HTML-to-text conversion often
    # flattens ``POLICY\nTanger ...`` into ``POLICY Tanger ...``. Keep the alias
    # eligible for hard control instead of leaving a spurious SOFT occurrence.
    if (
        left is not None
        and _is_all_caps_heading_token(left["token"])
        and _looks_like_entity_alias(source_term)
    ):
        return {
            "usage_status": "DEFINED_TERM_USAGE",
            "control": "HARD",
            "confidence": 0.98,
            "definition_role": "OTHER",
            "reasons": ["ALL_CAPS_HEADING_BOUNDARY"],
            "compound_context": _join_context(left, surface, None),
            "context_classification": "HEADING_BOUNDARY",
        }

    entity_name = _entity_full_name_context(source_term, surface, left, right)
    if entity_name is not None:
        return {
            "usage_status": "OTHER_NAMED_EXPRESSION_COMPONENT",
            "control": "NONE",
            "confidence": 0.98,
            "definition_role": "OTHER_NAME_COMPONENT",
            "reasons": ["COMPONENT_OF_ENTITY_FULL_NAME"],
            "compound_context": entity_name,
            "context_classification": "ENTITY_FULL_NAME",
        }

    relation = _entity_relation_context(source_term, surface, left, right)
    if relation is not None:
        return {
            "usage_status": "ENTITY_RELATION_USAGE",
            "control": "HARD",
            "confidence": 0.97,
            "definition_role": "OTHER",
            "reasons": ["ENTITY_ALIAS_AS_RELATION_MODIFIER"],
            "compound_context": relation,
            "context_classification": "ENTITY_RELATION",
        }

    if (
        left is not None
        and left["token"].casefold() in _SAFE_EXTERNAL_MODIFIERS
        and (_looks_like_acronym(source_term) or _is_multiword_term(source_term))
    ):
        return {
            "usage_status": "EXTERNAL_MODIFIED_DEFINED_TERM",
            "control": "HARD",
            "confidence": 0.97,
            "definition_role": "OTHER",
            "reasons": ["SAFE_EXTERNAL_MODIFIER"],
            "compound_context": _join_context(left, surface, None),
            "context_classification": "EXTERNAL_SAFE_MODIFIER",
        }

    named_head = _generic_named_head_context(source_term, surface, left, right)
    if named_head is not None:
        return {
            "usage_status": "OTHER_NAMED_EXPRESSION_COMPONENT",
            "control": "NONE",
            "confidence": 0.96,
            "definition_role": "OTHER_NAME_COMPONENT",
            "reasons": ["COMPONENT_OF_NAMED_EXPRESSION"],
            "compound_context": named_head,
            "context_classification": "NAMED_EXPRESSION",
        }

    left_capitalised = left is not None and _is_capitalised_token(left["token"])
    right_capitalised = right is not None and _is_capitalised_token(right["token"])
    if not left_capitalised and not right_capitalised:
        return None
    return {
        "usage_status": "POSSIBLY_NESTED_USAGE",
        "control": "SOFT",
        "confidence": 0.72,
        "definition_role": "POSSIBLY_NESTED",
        "reasons": ["ADJACENT_CAPITALISED_MODIFIER"],
        "compound_context": _join_context(
            left if left_capitalised else None,
            surface,
            right if right_capitalised else None,
        ),
        "context_classification": "AMBIGUOUS_CAPITALISED_COMPOUND",
    }


def _adjacent_context(paragraph: str, start: int, end: int) -> dict[str, dict[str, Any] | None]:
    """Return immediately adjacent lexical tokens without crossing punctuation."""
    left_offset = max(0, start - 100)
    left_text = paragraph[left_offset:start]
    right_text = paragraph[end:min(len(paragraph), end + 100)]
    left_matches = list(_WORD_RE.finditer(left_text))
    right_match = _WORD_RE.search(right_text)

    left: dict[str, Any] | None = None
    if left_matches:
        candidate = left_matches[-1]
        gap = left_text[candidate.end():]
        if re.fullmatch(r"[\s\-–—/]*", gap):
            left = {
                "token": candidate.group(0),
                "start": left_offset + candidate.start(),
                "end": left_offset + candidate.end(),
                "gap": gap,
            }

    right: dict[str, Any] | None = None
    if right_match is not None:
        gap = right_text[:right_match.start()]
        if re.fullmatch(r"[\s\-–—/]*", gap):
            right = {
                "token": right_match.group(0),
                "start": end + right_match.start(),
                "end": end + right_match.end(),
                "gap": gap,
            }
    return {"left": left, "right": right}


def _containing_job_title(paragraph: str, start: int, end: int) -> str | None:
    for match in _JOB_TITLE_PATTERN.finditer(paragraph):
        if match.start() <= start and end <= match.end() and (match.start(), match.end()) != (start, end):
            return match.group(0)
    return None


def _containing_legal_instrument_name(
    paragraph: str,
    source_term: str,
    start: int,
    end: int,
    context: dict[str, dict[str, Any] | None],
) -> str | None:
    right = context.get("right")
    if right is None or right["token"].casefold() not in _LEGAL_INSTRUMENT_HEADS:
        return None
    # If the source term already includes the same head, the adjacent token is not
    # evidence that this occurrence belongs to another instrument name.
    source_tokens = {x.casefold() for x in _WORD_RE.findall(source_term)}
    if right["token"].casefold() in source_tokens:
        return None
    phrase_start = start
    left = context.get("left")
    if left is not None and _is_capitalised_token(left["token"]):
        phrase_start = left["start"]
    return paragraph[phrase_start:right["end"]]


def _entity_full_name_context(
    source_term: str,
    surface: str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> str | None:
    if right is not None and right["token"].casefold() in _ENTITY_FULL_NAME_SUFFIXES:
        return _join_context(left if left and _is_capitalised_token(left["token"]) else None, surface, right)
    if (
        source_term.casefold() in _ENTITY_FULL_NAME_SUFFIXES
        and left is not None
        and _is_capitalised_token(left["token"])
        and not _is_possessive_token(left["token"])
    ):
        return _join_context(left, surface, None)
    return None


def _entity_relation_context(
    source_term: str,
    surface: str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> str | None:
    # A capitalised alias followed by an organisational relationship noun can keep
    # the alias under hard control: Gilead Leadership, Tanger Team, etc.
    if right is not None and right["token"].casefold() in _ENTITY_RELATION_NOUNS:
        if _looks_like_entity_alias(source_term):
            return _join_context(None, surface, right)
    return None


def _generic_named_head_context(
    source_term: str,
    surface: str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> str | None:
    if source_term.casefold() not in _GENERIC_NAMED_HEADS:
        return None
    if left is None or not _is_capitalised_token(left["token"]) or _is_possessive_token(left["token"]):
        return None
    return _join_context(left, surface, right if right and _is_capitalised_token(right["token"]) else None)


def _join_context(
    left: dict[str, Any] | None,
    surface: str,
    right: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    if left is not None:
        parts.append(left["token"])
    parts.append(surface)
    if right is not None:
        parts.append(right["token"])
    return " ".join(parts)


def _is_possessive_token(token: str) -> bool:
    folded = token.casefold()
    return folded.endswith("'s") or folded.endswith("’s")


def _has_possessive_suffix(paragraph: str, end: int) -> bool:
    return bool(re.match(r"(?:'s|’s)\b", paragraph[end:]))


def _is_all_caps_heading_token(token: str | None) -> bool:
    if not token:
        return False
    letters = re.sub(r"[^A-Za-z]", "", token)
    return (
        len(letters) >= 3
        and letters.isupper()
        and token.casefold() in _ALL_CAPS_HEADING_TOKENS
    )


def _is_capitalised_token(token: str | None) -> bool:
    if not token:
        return False
    if token.casefold() in _DETERMINER_OR_POSSESSIVE_MODIFIERS:
        return False
    letters = re.sub(r"[^A-Za-z]", "", token)
    return bool(letters) and (letters.isupper() or token[0].isupper())


def _looks_like_acronym(source_term: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", source_term)
    return len(letters) >= 2 and letters.isupper()


def _is_multiword_term(source_term: str) -> bool:
    return len(_WORD_RE.findall(source_term)) >= 2


def _looks_like_entity_alias(source_term: str) -> bool:
    tokens = _WORD_RE.findall(source_term)
    if not tokens:
        return False
    if len(tokens) > 1:
        return False
    folded = source_term.casefold()
    if folded in {
        "person", "party", "board", "plan", "policy", "program", "bank", "code", "act",
        "executive", "employee", "employer", "company", "committee", "agreement", "bonus",
        "salary", "payment", "period", "reason", "cause", "retirement", "disability",
    }:
        return False
    return source_term[:1].isupper() or _looks_like_acronym(source_term)


def _finish(
    result: dict[str, Any],
    *,
    surface: str,
    usage_status: str,
    control: str,
    confidence: float,
    definition_role: str,
    reasons: list[str],
    compound_context: str | None = None,
    context_classification: str | None = None,
) -> dict[str, Any]:
    if control not in CONTROL_LEVELS:
        raise ValueError(control)
    if usage_status not in USAGE_STATUSES:
        raise ValueError(usage_status)
    result["surface"] = surface
    result["usage_status"] = usage_status
    result["control_eligibility"] = control
    result["validation_confidence"] = confidence
    result["definition_role"] = definition_role
    result["validation_reasons"] = reasons
    result["compound_context"] = compound_context
    result["context_classification"] = context_classification
    result["validator_version"] = VALIDATOR_VERSION
    return result


def document_signature(document: dict[str, Any]) -> str:
    """Return the stable source signature used by validation and controlled translation."""
    return _document_signature(document)


def _document_signature(document: dict[str, Any]) -> str:
    payload = {
        "document_id": document.get("document_id"),
        "paragraphs": document.get("paragraphs", []),
        "benchmark_terms": document.get("benchmark_terms", []),
        "validator_version": VALIDATOR_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalise_surface(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _iter_documents(path: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if path.is_file():
        data = _load_json(path)
        if isinstance(data, dict) and "paragraphs" in data and "benchmark_terms" in data:
            yield path, data
        return
    if not path.exists():
        raise FileNotFoundError(path)
    for file in sorted(path.rglob("*.json")):
        if file.name.endswith("manifest.json"):
            continue
        data = _load_json(file)
        if isinstance(data, dict) and "paragraphs" in data and "benchmark_terms" in data:
            yield file, data


def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "document"
