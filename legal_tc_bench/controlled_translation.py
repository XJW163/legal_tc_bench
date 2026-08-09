from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import httpx
from tqdm import tqdm

from .io_utils import atomic_write_json, exclusive_process_lock
from .occurrence_validation import (
    VALIDATOR_VERSION,
    document_signature,
    load_occurrence_validations,
    occurrence_map_by_paragraph,
    validate_document_occurrences,
)
from .translate import (
    LATIN_LETTER_RE,
    _build_previous_context,
    _clean_translation,
    _safe_slug,
    _text_hash,
    contains_latin_letters,
    find_latin_residues,
)

PLACEHOLDER_RE = re.compile(r"【术语\d{4,}(?:_\d{2,4})?】")
PLACEHOLDER_STRATEGY_VERSION = "occurrence-unique-retry-quality-gate-v6"

CONTROLLED_SYSTEM_PROMPT = """You are a professional English-to-Simplified-Chinese legal translator operating under mandatory document-level terminology controls.
Translate only the current paragraph, accurately and completely.
Preserve every placeholder such as 【术语0001_01】 exactly, character-for-character. Each placeholder is a unique occurrence identifier and must appear exactly once in the output. Never translate, delete, duplicate, merge, split, or alter a placeholder. You may move a complete placeholder only when Chinese word order requires it.
The application will replace each placeholder with its approved Chinese legal term after generation. Therefore do not write the approved term beside the placeholder and do not output bilingual text.
Translate every other English expression into professional Simplified Chinese. Preserve Arabic numbers, dates, amounts, punctuation, section hierarchy, cross-references, and [***]. Do not summarize, omit, explain, or add content. Return only the translation of the current paragraph.
"""


PLACEHOLDER_REPAIR_SYSTEM_PROMPT = """You repair a Simplified-Chinese legal translation whose protected terminology placeholders were lost or duplicated.
Return the complete corrected Chinese translation only.
Every placeholder in the required inventory is a unique source-term occurrence and must appear exactly once, character-for-character. Do not invent any other placeholder. Do not translate a placeholder or place its approved Chinese term beside it.
Preserve all substantive meaning, numbers, dates, amounts, cross-references, punctuation and [***]. Make only the minimum wording changes needed to restore the placeholder inventory and complete any content omitted with a missing placeholder.
This legacy repair mode is available only for ablation. The default policy regenerates from the protected source instead of editing a prior Chinese output.
"""

SEMANTIC_VERIFIER_SYSTEM_PROMPT = """You are a strict bilingual legal-translation verifier.
Compare one English source paragraph with one Simplified-Chinese candidate translation.
Judge only fidelity and protected-term attachment. Reject any unsupported addition, meaning omission, mistranslation, duplicated term wording, unnatural attachment of a protected term, broken legal logic, or unresolved/vague reference introduced by translation.
The candidate is required to be fully translated into Simplified Chinese. Every listed English protected term MUST use the listed Chinese canonical rendering. Never require a protected term to remain in English, and never mark a candidate wrong merely because it correctly uses a listed Chinese canonical rendering. However, if two different English terms have been assigned the same Chinese rendering and that collapses a legally meaningful distinction, report that problem explicitly.
The listed canonical terms are mandatory renderings, but their mere presence is not sufficient for a PASS.
Return one JSON object only, with exactly these keys:
{
  "adequacy": "PASS" or "FAIL",
  "term_semantics": "PASS" or "FAIL",
  "unsupported_addition": true or false,
  "meaning_omission": true or false,
  "unnatural_attachment": true or false,
  "broken_legal_logic": true or false,
  "problematic_terms": [strings],
  "rationale": "brief Chinese explanation"
}
"""


class PlaceholderMismatchError(ValueError):
    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(
            "placeholder mismatch: "
            f"expected={diagnostics['expected']}, observed={diagnostics['observed']}, "
            f"missing={diagnostics['missing']}, duplicated={diagnostics['duplicated']}, "
            f"unknown={diagnostics['unknown']}"
        )


class PlaceholderQualityError(ValueError):
    def __init__(self, stage: str, diagnostics: dict[str, Any]):
        self.stage = stage
        self.diagnostics = diagnostics
        super().__init__(f"placeholder quality gate rejected output at {stage}: {diagnostics}")


class ControlledTranslationTruncatedError(ValueError):
    """Raised when the model stops because the output-token budget was exhausted."""

GLOSSARY_SYSTEM_PROMPT = """You are a professional English-to-Simplified-Chinese legal translator operating under document-level terminology controls.
Translate only the current paragraph, accurately and completely.
Terms in the mandatory-control section must use the approved Chinese rendering whenever that validated defined-term occurrence appears. Do not substitute synonyms or retain the English term.
Terms in the conditional-hint section are ambiguous candidates: use the approved rendering only when the local context truly denotes the contract-defined term. Do not force it when the string is a common word or part of another legal name.
Translate all other content into professional Simplified Chinese. Preserve Arabic numbers, dates, amounts, punctuation, section hierarchy, cross-references, and [***]. Do not summarize, omit, explain, or add content. Return only the translation of the current paragraph.
"""


def controlled_batch_translate(
    annotated: Path,
    term_memory: Path,
    output_root: Path,
    *,
    experiment_name: str,
    model: str,
    base_url: str,
    constraint_mode: str = "placeholder",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    context_paragraphs: int = 2,
    max_context_chars: int = 12000,
    max_retries: int = 5,
    timeout_seconds: float = 600.0,
    workers: int = 1,
    overwrite: bool = False,
    continue_on_error: bool = False,
    max_documents: int | None = None,
    document_ids: set[str] | None = None,
    occurrence_validation: Path | None = None,
    placeholder_repair_policy: str = "retry_then_fallback",
    placeholder_quality_gate: bool = True,
    placeholder_semantic_verifier: str = "retry",
    placeholder_repeat_rejection_limit: int = 2,
    semantic_verifier_retries: int = 2,
) -> dict[str, Any]:
    """Translate documents with document-specific legal term controls.

    ``placeholder`` mode gives a hard lexical guarantee when the model preserves all
    placeholders. ``glossary`` mode is retained as an ablation baseline and relies on
    prompt adherence only.
    """
    if constraint_mode not in {"placeholder", "glossary"}:
        raise ValueError("constraint_mode must be placeholder or glossary")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if placeholder_repair_policy not in {"retry_then_fallback", "llm_repair"}:
        raise ValueError("placeholder_repair_policy must be retry_then_fallback or llm_repair")
    if placeholder_semantic_verifier not in {"off", "retry", "all"}:
        raise ValueError("placeholder_semantic_verifier must be off, retry, or all")
    if placeholder_repeat_rejection_limit < 0:
        raise ValueError("placeholder_repeat_rejection_limit must be >= 0")
    if semantic_verifier_retries < 1:
        raise ValueError("semantic_verifier_retries must be >= 1")

    annotated = Path(annotated)
    term_memory = Path(term_memory)
    experiment_dir = Path(output_root) / _safe_slug(experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    memories = _load_memories(term_memory)
    occurrence_validation = Path(occurrence_validation) if occurrence_validation else None
    validations = load_occurrence_validations(occurrence_validation) if occurrence_validation else {}
    documents = list(_iter_documents(annotated))
    if document_ids:
        documents = [x for x in documents if str(x[1].get("document_id") or x[0].stem) in document_ids]
    if max_documents is not None:
        if max_documents < 1:
            raise ValueError("max_documents must be >= 1")
        documents = documents[:max_documents]
    manifest_path = experiment_dir / "controlled_translation_manifest.json"
    manifest = _load_json(manifest_path, {})
    manifest.update({
        "schema_version": "1.4",
        "status": "in_progress",
        "experiment": experiment_name,
        "placeholder_strategy_version": PLACEHOLDER_STRATEGY_VERSION if constraint_mode == "placeholder" else None,
        "model": model,
        "base_url": base_url,
        "constraint_mode": constraint_mode,
        "placeholder_repair_policy": placeholder_repair_policy if constraint_mode == "placeholder" else None,
        "placeholder_quality_gate": bool(placeholder_quality_gate) if constraint_mode == "placeholder" else None,
        "placeholder_semantic_verifier": placeholder_semantic_verifier if constraint_mode == "placeholder" else None,
        "placeholder_repeat_rejection_limit": placeholder_repeat_rejection_limit if constraint_mode == "placeholder" else None,
        "semantic_verifier_retries": semantic_verifier_retries if constraint_mode == "placeholder" else None,
        "annotated": str(annotated),
        "term_memory": str(term_memory),
        "occurrence_validation": str(occurrence_validation) if occurrence_validation else None,
        "occurrence_validator_version": VALIDATOR_VERSION,
        "documents": manifest.get("documents", {}),
        "total_documents": len(documents),
        "updated_at": _now(),
    })
    atomic_write_json(manifest_path, manifest)

    totals = {
        "documents_total": len(documents),
        "documents_complete": 0,
        "documents_incomplete": 0,
        "paragraphs_total": 0,
        "paragraphs_translated": 0,
        "paragraphs_restored": 0,
        "paragraph_errors": 0,
        "controlled_term_occurrences": 0,
        "hard_controlled_term_occurrences": 0,
        "soft_controlled_term_occurrences": 0,
        "excluded_term_occurrences": 0,
        "placeholder_retries": 0,
        "generation_retries": 0,
        "glossary_compliance_retries": 0,
        "placeholder_regeneration_attempts": 0,
        "placeholder_legacy_repair_attempts": 0,
        "placeholder_quality_gate_rejections": 0,
        "placeholder_inventory_rejections": 0,
        "placeholder_context_rejections": 0,
        "placeholder_restored_rejections": 0,
        "placeholder_semantic_verifier_calls": 0,
        "placeholder_semantic_rejections": 0,
        "placeholder_request_or_processing_rejections": 0,
        "placeholder_early_fallback_paragraphs": 0,
        "placeholder_contextual_omission_paragraphs": 0,
        "placeholder_contextual_omission_occurrences": 0,
        "placeholder_fallback_verified_paragraphs": 0,
        "unresolved_paragraphs": 0,
        "definition_alias_normalized_paragraphs": 0,
        "definition_alias_normalized_occurrences": 0,
        "legitimate_definition_repetition_paragraphs": 0,
        "legitimate_definition_repetition_pairs": 0,
        "max_tokens_escalations": 0,
        "placeholder_exact_paragraphs": 0,
        "placeholder_retry_success_paragraphs": 0,
        "placeholder_repaired_paragraphs": 0,
        "conditional_glossary_paragraphs": 0,
        "uncontrolled_paragraphs": 0,
        "placeholder_fallback_paragraphs": 0,
        "placeholder_fallback_noncompliance_paragraphs": 0,
        "glossary_noncompliance_paragraphs": 0,
        "errors": 0,
    }

    def process(path: Path, doc: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
        document_id = str(doc.get("document_id") or path.stem)
        memory = memories.get(document_id)
        if memory is None:
            raise ValueError(f"missing term memory for document {document_id}")
        validation = validations.get(document_id)
        if occurrence_validation is not None and validation is None:
            raise ValueError(f"missing occurrence validation for document {document_id}")
        destination = experiment_dir / f"{_safe_slug(document_id)}.controlled.translation.json"
        result = _controlled_translate_document(
            doc,
            source_path=path,
            memory=memory,
            occurrence_validation=validation,
            output_path=destination,
            model=model,
            base_url=base_url,
            experiment_name=experiment_name,
            constraint_mode=constraint_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            context_paragraphs=context_paragraphs,
            max_context_chars=max_context_chars,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
            show_progress=(workers == 1),
            placeholder_repair_policy=placeholder_repair_policy,
            placeholder_quality_gate=placeholder_quality_gate,
            placeholder_semantic_verifier=placeholder_semantic_verifier,
            placeholder_repeat_rejection_limit=placeholder_repeat_rejection_limit,
            semantic_verifier_retries=semantic_verifier_retries,
        )
        return document_id, destination, result

    def record_success(source: Path, document_id: str, destination: Path, result: dict[str, Any]) -> None:
        stats = result["statistics"]
        totals["paragraphs_total"] += int(stats["total_paragraphs"])
        totals["paragraphs_translated"] += int(stats["translated_paragraphs"])
        totals["paragraphs_restored"] += int(stats.get("restored_paragraphs", 0))
        totals["paragraph_errors"] += int(stats.get("failed_paragraphs", 0))
        totals["controlled_term_occurrences"] += int(stats.get("controlled_term_occurrences", 0))
        totals["hard_controlled_term_occurrences"] += int(stats.get("hard_controlled_term_occurrences", 0))
        totals["soft_controlled_term_occurrences"] += int(stats.get("soft_controlled_term_occurrences", 0))
        totals["excluded_term_occurrences"] += int(stats.get("excluded_term_occurrences", 0))
        totals["placeholder_retries"] += int(stats.get("placeholder_retries", 0))
        totals["generation_retries"] += int(stats.get("generation_retries", 0))
        totals["glossary_compliance_retries"] += int(stats.get("glossary_compliance_retries", 0))
        totals["placeholder_regeneration_attempts"] += int(stats.get("placeholder_regeneration_attempts", 0))
        totals["placeholder_legacy_repair_attempts"] += int(stats.get("placeholder_legacy_repair_attempts", 0))
        totals["placeholder_quality_gate_rejections"] += int(stats.get("placeholder_quality_gate_rejections", 0))
        totals["placeholder_inventory_rejections"] += int(stats.get("placeholder_inventory_rejections", 0))
        totals["placeholder_context_rejections"] += int(stats.get("placeholder_context_rejections", 0))
        totals["placeholder_restored_rejections"] += int(stats.get("placeholder_restored_rejections", 0))
        totals["placeholder_semantic_verifier_calls"] += int(stats.get("placeholder_semantic_verifier_calls", 0))
        totals["placeholder_semantic_rejections"] += int(stats.get("placeholder_semantic_rejections", 0))
        totals["placeholder_request_or_processing_rejections"] += int(stats.get("placeholder_request_or_processing_rejections", 0))
        totals["placeholder_early_fallback_paragraphs"] += int(stats.get("placeholder_early_fallback_paragraphs", 0))
        totals["placeholder_contextual_omission_paragraphs"] += int(stats.get("placeholder_contextual_omission_paragraphs", 0))
        totals["placeholder_contextual_omission_occurrences"] += int(stats.get("placeholder_contextual_omission_occurrences", 0))
        totals["definition_alias_normalized_paragraphs"] += int(stats.get("definition_alias_normalized_paragraphs", 0))
        totals["definition_alias_normalized_occurrences"] += int(stats.get("definition_alias_normalized_occurrences", 0))
        totals["legitimate_definition_repetition_paragraphs"] += int(stats.get("legitimate_definition_repetition_paragraphs", 0))
        totals["legitimate_definition_repetition_pairs"] += int(stats.get("legitimate_definition_repetition_pairs", 0))
        totals["max_tokens_escalations"] += int(stats.get("max_tokens_escalations", 0))
        totals["placeholder_exact_paragraphs"] += int(stats.get("placeholder_exact_paragraphs", 0))
        totals["placeholder_retry_success_paragraphs"] += int(stats.get("placeholder_retry_success_paragraphs", 0))
        totals["placeholder_repaired_paragraphs"] += int(stats.get("placeholder_repaired_paragraphs", 0))
        totals["conditional_glossary_paragraphs"] += int(stats.get("conditional_glossary_paragraphs", 0))
        totals["uncontrolled_paragraphs"] += int(stats.get("uncontrolled_paragraphs", 0))
        totals["placeholder_fallback_paragraphs"] += int(stats.get("placeholder_fallback_paragraphs", 0))
        totals["placeholder_fallback_noncompliance_paragraphs"] += int(stats.get("placeholder_fallback_noncompliance_paragraphs", 0))
        totals["glossary_noncompliance_paragraphs"] += int(stats.get("glossary_noncompliance_paragraphs", 0))
        if result["status"] == "complete":
            totals["documents_complete"] += 1
        else:
            totals["documents_incomplete"] += 1
        manifest["documents"][document_id] = {
            "source": str(source),
            "output": str(destination),
            "status": result["status"],
            "statistics": stats,
            "updated_at": result["updated_at"],
        }
        totals["placeholder_fallback_verified_paragraphs"] += int(stats.get("placeholder_fallback_verified_paragraphs", 0))
        totals["unresolved_paragraphs"] += int(stats.get("unresolved_paragraphs", 0))

    def record_failure(source: Path, doc: dict[str, Any], exc: Exception) -> None:
        document_id = str(doc.get("document_id") or source.stem)
        totals["errors"] += 1
        totals["documents_incomplete"] += 1
        manifest["documents"][document_id] = {
            "source": str(source),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at": _now(),
        }
        tqdm.write(f"controlled translation failed for {document_id}: {exc}")

    lock_path = experiment_dir / ".controlled-translate.lock"
    with exclusive_process_lock(lock_path, label=f"controlled-translate:{experiment_name}"):
        progress = tqdm(total=len(documents), desc=f"controlled {experiment_name}", unit="doc", dynamic_ncols=True)
        try:
            if workers == 1:
                for source, doc in documents:
                    try:
                        document_id, destination, result = process(source, doc)
                        record_success(source, document_id, destination, result)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        record_failure(source, doc, exc)
                        if not continue_on_error:
                            raise
                    finally:
                        progress.update(1)
                        manifest["totals"] = totals
                        manifest["updated_at"] = _now()
                        atomic_write_json(manifest_path, manifest)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="legal-tc-controlled") as executor:
                    futures: dict[Future[Any], tuple[Path, dict[str, Any]]] = {
                        executor.submit(process, source, doc): (source, doc)
                        for source, doc in documents
                    }
                    first_error: Exception | None = None
                    for future in as_completed(futures):
                        source, doc = futures[future]
                        try:
                            document_id, destination, result = future.result()
                            record_success(source, document_id, destination, result)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            record_failure(source, doc, exc)
                            if first_error is None:
                                first_error = exc
                        finally:
                            progress.update(1)
                            manifest["totals"] = totals
                            manifest["updated_at"] = _now()
                            atomic_write_json(manifest_path, manifest)
                    if first_error is not None and not continue_on_error:
                        raise first_error
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["totals"] = totals
            manifest["updated_at"] = _now()
            atomic_write_json(manifest_path, manifest)
            raise
        finally:
            progress.close()

    # manifest["status"] = "complete" if totals["documents_complete"] == len(documents) else "partial"
    if totals["documents_incomplete"] > 0:
        manifest["status"] = "partial"
    elif totals["unresolved_paragraphs"] > 0:
        manifest["status"] = "completed_with_unresolved"
    else:
        manifest["status"] = "complete"
    manifest["totals"] = totals
    manifest["updated_at"] = _now()
    atomic_write_json(manifest_path, manifest)
    return {"experiment_directory": str(experiment_dir), "manifest": str(manifest_path), "status": manifest["status"], **totals}


def inject_term_placeholders(
    source_text: str,
    occurrences: list[dict[str, Any]],
    memory_by_term: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Replace occurrence spans with stable occurrence-unique placeholders.

    Reusing one token for every occurrence forces the model to count repeated identical
    strings, which is fragile in long legal paragraphs.  Each selected occurrence now
    receives its own token (for example ``【术语0002_03】``), while every token still
    restores to the same document-level canonical translation.
    """
    candidates: list[dict[str, Any]] = []
    for occ in occurrences:
        term_id = str(occ.get("term_id"))
        memory = memory_by_term.get(term_id)
        if not memory or not str(memory.get("canonical_translation", "")).strip():
            continue
        start = int(occ.get("char_start"))
        end = int(occ.get("char_end"))
        if start < 0 or end <= start or end > len(source_text):
            continue
        candidates.append({
            "term_id": term_id,
            "source_term": str(occ.get("source_term", "")),
            "char_start": start,
            "char_end": end,
            "surface": source_text[start:end],
            "canonical_translation": str(memory["canonical_translation"]),
            "legal_scope_summary": str(memory.get("legal_scope_summary", "")),
            "criticality": str(memory.get("criticality", "")),
            "allowed_variants": memory.get("allowed_variants", []),
            "risky_variants": memory.get("risky_variants", []),
            "subsequent_short_forms": memory.get("subsequent_short_forms", []),
            "control_eligibility": str(occ.get("control_eligibility") or "HARD").upper(),
            "usage_status": occ.get("usage_status"),
            "validation_reasons": occ.get("validation_reasons", []),
            "compound_context": occ.get("compound_context"),
            "context_classification": occ.get("context_classification"),
        })

    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for item in sorted(candidates, key=lambda x: (-(x["char_end"] - x["char_start"]), x["char_start"])):
        span = (item["char_start"], item["char_end"])
        if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
            continue
        selected.append(item)
        occupied.append(span)

    selected.sort(key=lambda x: x["char_start"])
    totals: dict[str, int] = {}
    for item in selected:
        totals[item["term_id"]] = totals.get(item["term_id"], 0) + 1
    running: dict[str, int] = {}
    for item in selected:
        term_id = item["term_id"]
        running[term_id] = running.get(term_id, 0) + 1
        base = _placeholder_for_term(term_id)
        if totals[term_id] == 1:
            placeholder = base
        else:
            placeholder = base[:-1] + f"_{running[term_id]:02d}】"
        item["base_placeholder"] = base
        item["placeholder"] = placeholder
        item["occurrence_index"] = running[term_id]
        item["occurrence_count_for_term"] = totals[term_id]

    protected = source_text
    for item in sorted(selected, key=lambda x: x["char_start"], reverse=True):
        protected = protected[:item["char_start"]] + item["placeholder"] + protected[item["char_end"]:]
    return protected, selected


def prepare_term_controls(
    source_text: str,
    occurrences: list[dict[str, Any]],
    memory_by_term: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare HARD placeholder controls, SOFT conditional hints, and exclusions.

    Old datasets without occurrence-validation fields remain backward compatible and
    default to HARD.  Validated data should always provide ``control_eligibility``.
    """
    hard_occurrences: list[dict[str, Any]] = []
    soft_occurrences: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for occurrence in occurrences:
        eligibility = str(occurrence.get("control_eligibility") or "HARD").upper()
        if eligibility == "HARD":
            hard_occurrences.append(occurrence)
        elif eligibility == "SOFT":
            soft_occurrences.append(occurrence)
        else:
            excluded.append(dict(occurrence))

    protected, hard_constraints = inject_term_placeholders(
        source_text,
        hard_occurrences,
        memory_by_term,
    )
    hard_spans = {(x["char_start"], x["char_end"]) for x in hard_constraints}
    soft_constraints: list[dict[str, Any]] = []
    for occurrence in soft_occurrences:
        term_id = str(occurrence.get("term_id"))
        memory = memory_by_term.get(term_id)
        if not memory or not str(memory.get("canonical_translation", "")).strip():
            excluded.append(dict(occurrence))
            continue
        try:
            start = int(occurrence.get("char_start"))
            end = int(occurrence.get("char_end"))
        except (TypeError, ValueError):
            excluded.append(dict(occurrence))
            continue
        if start < 0 or end <= start or end > len(source_text):
            excluded.append(dict(occurrence))
            continue
        if any(not (end <= a or start >= b) for a, b in hard_spans):
            excluded.append(dict(occurrence))
            continue
        soft_constraints.append({
            "term_id": term_id,
            "source_term": str(occurrence.get("source_term", "")),
            "char_start": start,
            "char_end": end,
            "surface": source_text[start:end],
            "canonical_translation": str(memory["canonical_translation"]),
            "legal_scope_summary": str(memory.get("legal_scope_summary", "")),
            "criticality": str(memory.get("criticality", "")),
            "allowed_variants": memory.get("allowed_variants", []),
            "risky_variants": memory.get("risky_variants", []),
            "subsequent_short_forms": memory.get("subsequent_short_forms", []),
            "control_eligibility": "SOFT",
            "usage_status": occurrence.get("usage_status"),
            "validation_reasons": occurrence.get("validation_reasons", []),
            "compound_context": occurrence.get("compound_context"),
            "context_classification": occurrence.get("context_classification"),
            "placeholder": None,
            "base_placeholder": None,
            "occurrence_index": occurrence.get("occurrence_index"),
        })
    return protected, hard_constraints, soft_constraints, excluded


def _controlled_translate_document(
    dataset: dict[str, Any],
    *,
    source_path: Path,
    memory: dict[str, Any],
    occurrence_validation: dict[str, Any] | None,
    output_path: Path,
    model: str,
    base_url: str,
    experiment_name: str,
    constraint_mode: str,
    temperature: float,
    max_tokens: int,
    context_paragraphs: int,
    max_context_chars: int,
    max_retries: int,
    timeout_seconds: float,
    overwrite: bool,
    show_progress: bool,
    placeholder_repair_policy: str,
    placeholder_quality_gate: bool,
    placeholder_semantic_verifier: str,
    placeholder_repeat_rejection_limit: int,
    semantic_verifier_retries: int,
) -> dict[str, Any]:
    document_id = str(dataset.get("document_id") or source_path.stem)
    paragraphs = dataset.get("paragraphs") or []
    memory_by_term = {str(x.get("term_id")): x for x in memory.get("terms", [])}
    memory_collision_warnings = _term_memory_collision_warnings(memory.get("terms", []))
    memory_signature = _json_hash(memory.get("terms", []))
    if occurrence_validation is None:
        occurrence_validation = validate_document_occurrences(dataset, source_path=source_path)
        occurrence_validation_source = "inline-deterministic"
    else:
        occurrence_validation_source = "precomputed"
        expected_source_signature = document_signature(dataset)
        actual_source_signature = str(occurrence_validation.get("source_signature") or "")
        if actual_source_signature != expected_source_signature:
            raise ValueError(
                f"stale occurrence validation for {document_id}: source signature changed; "
                "rerun validate-term-occurrences"
            )
    validation_signature = _json_hash(occurrence_validation.get("terms", []))
    control_signature = _json_hash({
        "model": model,
        "base_url": base_url,
        "constraint_mode": constraint_mode,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "context_paragraphs": context_paragraphs,
        "max_context_chars": max_context_chars,
        "memory_signature": memory_signature,
        "occurrence_validation_signature": validation_signature,
        "occurrence_validator_version": occurrence_validation.get("validator_version") or VALIDATOR_VERSION,
        "placeholder_strategy_version": PLACEHOLDER_STRATEGY_VERSION if constraint_mode == "placeholder" else None,
        "placeholder_repair_policy": placeholder_repair_policy if constraint_mode == "placeholder" else None,
        "placeholder_quality_gate": bool(placeholder_quality_gate) if constraint_mode == "placeholder" else None,
        "placeholder_semantic_verifier": placeholder_semantic_verifier if constraint_mode == "placeholder" else None,
        "placeholder_repeat_rejection_limit": placeholder_repeat_rejection_limit if constraint_mode == "placeholder" else None,
        "semantic_verifier_retries": semantic_verifier_retries if constraint_mode == "placeholder" else None,
    })

    occurrence_by_pid = occurrence_map_by_paragraph(occurrence_validation)

    existing = {} if overwrite else _load_json(output_path, {})
    if existing and existing.get("control_signature") != control_signature:
        raise ValueError(
            f"controlled translation settings or term memory changed for {output_path}; "
            "use a new experiment/output or --overwrite"
        )
    restored: dict[str, dict[str, Any]] = {}
    for item in existing.get("translations", []) if isinstance(existing, dict) else []:
        pid = str(item.get("paragraph_id"))
        source = str(item.get("source", ""))
        if (
            str(item.get("translation", "")).strip()
            and item.get("source_hash") == _text_hash(source)
            and item.get("control_signature") == control_signature
        ):
            restored[pid] = item

    state = {
        "schema_version": "1.4",
        "document_id": document_id,
        "placeholder_strategy_version": PLACEHOLDER_STRATEGY_VERSION if constraint_mode == "placeholder" else None,
        "source_path": str(source_path),
        "experiment": experiment_name,
        "model": model,
        "base_url": base_url,
        "constraint_mode": constraint_mode,
        "placeholder_repair_policy": placeholder_repair_policy if constraint_mode == "placeholder" else None,
        "placeholder_quality_gate": bool(placeholder_quality_gate) if constraint_mode == "placeholder" else None,
        "placeholder_semantic_verifier": placeholder_semantic_verifier if constraint_mode == "placeholder" else None,
        "placeholder_repeat_rejection_limit": placeholder_repeat_rejection_limit if constraint_mode == "placeholder" else None,
        "semantic_verifier_retries": semantic_verifier_retries if constraint_mode == "placeholder" else None,
        "memory_signature": memory_signature,
        "occurrence_validation_signature": validation_signature,
        "occurrence_validation_source": occurrence_validation_source,
        "occurrence_validator_version": occurrence_validation.get("validator_version") or VALIDATOR_VERSION,
        "occurrence_validation_statistics": occurrence_validation.get("statistics", {}),
        "term_memory_collision_warnings": memory_collision_warnings,
        "control_signature": control_signature,
        "translations": [],
        "errors": existing.get("errors", []) if existing else [],
        "status": "in_progress",
        "started_at": existing.get("started_at") if existing else _now(),
        "updated_at": _now(),
    }

    from openai import OpenAI

    timeout = httpx.Timeout(
        connect=min(30.0, timeout_seconds),
        read=timeout_seconds,
        write=min(120.0, timeout_seconds),
        pool=min(30.0, timeout_seconds),
    )
    http_client = httpx.Client(trust_env=False, timeout=timeout)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )

    ordered: list[dict[str, Any]] = []
    restored_count = 0
    controlled_occurrences = 0
    hard_controlled_occurrences = 0
    soft_controlled_occurrences = 0
    excluded_occurrences = 0
    placeholder_retries = 0
    placeholder_repaired = 0
    placeholder_fallback = 0
    placeholder_fallback_noncompliance = 0
    glossary_noncompliance = 0
    errors: list[dict[str, Any]] = []
    try:
        progress = tqdm(paragraphs, desc=document_id, unit="para", leave=False, disable=not show_progress, dynamic_ncols=True)
        for index, paragraph in enumerate(progress):
            pid = str(paragraph.get("paragraph_id"))
            source_text = str(paragraph.get("text", ""))
            if pid in restored and restored[pid].get("source") == source_text:
                ordered.append(restored[pid])
                restored_count += 1
                continue

            protected, constraints, soft_constraints, excluded_controls = prepare_term_controls(
                source_text,
                occurrence_by_pid.get(pid, []),
                memory_by_term,
            )
            hard_controlled_occurrences += len(constraints)
            soft_controlled_occurrences += len(soft_constraints)
            excluded_occurrences += len(excluded_controls)
            controlled_occurrences += len(constraints) + len(soft_constraints)
            context = _build_previous_context(
                paragraphs=paragraphs,
                completed=ordered,
                current_index=index,
                count=context_paragraphs,
                max_chars=max_context_chars,
            )
            try:
                translation, model_output, attempts, latency, retry_count, compliance, control_meta = _call_controlled_translation(
                    client=client,
                    model=model,
                    source_text=source_text,
                    protected_source=protected,
                    constraints=constraints,
                    soft_constraints=soft_constraints,
                    previous_context=context,
                    constraint_mode=constraint_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    placeholder_repair_policy=placeholder_repair_policy,
                    placeholder_quality_gate=placeholder_quality_gate,
                    placeholder_semantic_verifier=placeholder_semantic_verifier,
                    placeholder_repeat_rejection_limit=placeholder_repeat_rejection_limit,
                    semantic_verifier_retries=semantic_verifier_retries,
                )
                placeholder_retries += retry_count
                if control_meta.get("strategy") == "placeholder_repaired":
                    placeholder_repaired += 1
                if control_meta.get("placeholder_fallback_used"):
                    placeholder_fallback += 1
                    if not compliance:
                        placeholder_fallback_noncompliance += 1
                if constraint_mode == "glossary" and not compliance:
                    glossary_noncompliance += 1
            except Exception as exc:
                errors.append({
                    "paragraph_id": paragraph.get("paragraph_id"),
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now(),
                })
                state["translations"] = ordered
                state["errors"] = errors
                state["statistics"] = _stats(paragraphs, ordered, restored_count, controlled_occurrences, hard_controlled_occurrences, soft_controlled_occurrences, excluded_occurrences, placeholder_retries, placeholder_repaired, placeholder_fallback, placeholder_fallback_noncompliance, glossary_noncompliance, errors)
                state["status"] = "incomplete"
                state["updated_at"] = _now()
                atomic_write_json(output_path, state)
                tqdm.write(f"controlled paragraph failed; continuing: {document_id} paragraph {pid}: {exc}")
                continue

            item = {
                "paragraph_id": paragraph.get("paragraph_id"),
                "source": source_text,
                "source_hash": _text_hash(source_text),
                "translation": translation,
                "model_output_before_restore": model_output if constraint_mode == "placeholder" else None,
                "model": model,
                "constraint_mode": constraint_mode,
                "control_signature": control_signature,
                "controlled_term_occurrence_count": len(constraints) + len(soft_constraints),
                "hard_controlled_term_occurrence_count": len(constraints),
                "soft_controlled_term_occurrence_count": len(soft_constraints),
                "excluded_term_occurrence_count": len(excluded_controls),
                "controlled_terms": [
                    {
                        "term_id": x["term_id"],
                        "source_term": x["source_term"],
                        "placeholder": x.get("placeholder") if constraint_mode == "placeholder" else None,
                        "base_placeholder": x.get("base_placeholder") if constraint_mode == "placeholder" else None,
                        "occurrence_index": x.get("occurrence_index"),
                        "canonical_translation": x["canonical_translation"],
                        "risky_variants": x.get("risky_variants", []),
                        "char_start": x["char_start"],
                        "char_end": x["char_end"],
                        "control_eligibility": x.get("control_eligibility", "HARD"),
                        "usage_status": x.get("usage_status"),
                        "validation_reasons": x.get("validation_reasons", []),
                        "compound_context": x.get("compound_context"),
                        "context_classification": x.get("context_classification"),
                    }
                    for x in (constraints + soft_constraints)
                ],
                "excluded_term_occurrences": [
                    {
                        "term_id": x.get("term_id"),
                        "source_term": x.get("source_term"),
                        "surface": x.get("surface"),
                        "char_start": x.get("char_start"),
                        "char_end": x.get("char_end"),
                        "usage_status": x.get("usage_status"),
                        "control_eligibility": x.get("control_eligibility", "NONE"),
                        "validation_reasons": x.get("validation_reasons", []),
                        "compound_context": x.get("compound_context"),
                        "context_classification": x.get("context_classification"),
                    }
                    for x in excluded_controls
                ],
                "constraint_compliance": compliance,
                "resolution_status": (
                    "glossary_fallback_noncompliant"
                    if control_meta.get("placeholder_fallback_used") and not compliance
                    else "glossary_fallback_verified"
                    if control_meta.get("placeholder_fallback_used")
                    else "verified"
                    if compliance
                    else "noncompliant"
                ),
                "glossary_compliance_details": control_meta.get("glossary_compliance_details"),
                "exact_glossary_compliance": control_meta.get("exact_glossary_compliance"),
                "attempts": attempts,
                "placeholder_retry_count": retry_count,
                "generation_retry_count": control_meta.get("generation_retries", retry_count),
                "glossary_compliance_retry_count": control_meta.get("glossary_compliance_retries", 0),
                "model_request_count": control_meta.get("model_request_count", attempts),
                "placeholder_strategy_version": control_meta.get("placeholder_strategy_version"),
                "placeholder_repair_policy": control_meta.get("placeholder_repair_policy"),
                "placeholder_quality_gate_enabled": control_meta.get("placeholder_quality_gate_enabled"),
                "placeholder_semantic_verifier": control_meta.get("placeholder_semantic_verifier"),
                "constraint_strategy_used": control_meta.get("strategy"),
                "placeholder_repair_attempts": control_meta.get("placeholder_repair_attempts", 0),
                "placeholder_regeneration_attempts": control_meta.get("placeholder_regeneration_attempts", 0),
                "placeholder_legacy_repair_attempts": control_meta.get("placeholder_legacy_repair_attempts", 0),
                "placeholder_quality_gate_rejections": control_meta.get("placeholder_quality_gate_rejections", []),
                "placeholder_inventory_diagnostics": control_meta.get("placeholder_inventory_diagnostics"),
                "placeholder_context_diagnostics": control_meta.get("placeholder_context_diagnostics"),
                "restored_translation_diagnostics": control_meta.get("restored_translation_diagnostics"),
                "placeholder_semantic_verifier_result": control_meta.get("placeholder_semantic_verifier_result"),
                "placeholder_semantic_verifier_calls": control_meta.get("placeholder_semantic_verifier_calls", 0),
                "placeholder_early_fallback": bool(control_meta.get("placeholder_early_fallback")),
                "placeholder_contextual_omission_tokens": control_meta.get("placeholder_contextual_omission_tokens", []),
                "placeholder_contextual_omission_count": control_meta.get("placeholder_contextual_omission_count", 0),
                "definition_alias_normalizations": control_meta.get("definition_alias_normalizations", []),
                "definition_alias_normalized_occurrence_count": control_meta.get("definition_alias_normalized_occurrence_count", 0),
                "legitimate_definition_repetitions": control_meta.get("legitimate_definition_repetitions", []),
                "legitimate_definition_repetition_count": control_meta.get("legitimate_definition_repetition_count", 0),
                "max_tokens_escalations": control_meta.get("max_tokens_escalations", 0),
                "placeholder_fallback_used": bool(control_meta.get("placeholder_fallback_used")),
                "placeholder_diagnostics": control_meta.get("placeholder_diagnostics"),
                "placeholder_failure": control_meta.get("placeholder_failure"),
                "latency_seconds": round(latency, 3),
                "latin_residue_count": len(LATIN_LETTER_RE.findall(translation)),
                "latin_residue_samples": find_latin_residues(translation),
                "latin_residue_warning": contains_latin_letters(translation),
                "completed_at": _now(),
            }
            ordered.append(item)
            errors = [x for x in errors if str(x.get("paragraph_id")) != pid]
            state["translations"] = ordered
            state["errors"] = errors
            state["statistics"] = _stats(paragraphs, ordered, restored_count, controlled_occurrences, hard_controlled_occurrences, soft_controlled_occurrences, excluded_occurrences, placeholder_retries, placeholder_repaired, placeholder_fallback, placeholder_fallback_noncompliance, glossary_noncompliance, errors)
            state["updated_at"] = _now()
            atomic_write_json(output_path, state)
    except KeyboardInterrupt:
        state["translations"] = ordered
        state["errors"] = errors
        state["statistics"] = _stats(paragraphs, ordered, restored_count, controlled_occurrences, hard_controlled_occurrences, soft_controlled_occurrences, excluded_occurrences, placeholder_retries, placeholder_repaired, placeholder_fallback, placeholder_fallback_noncompliance, glossary_noncompliance, errors)
        state["status"] = "interrupted"
        state["updated_at"] = _now()
        atomic_write_json(output_path, state)
        raise
    finally:
        http_client.close()

    state["translations"] = ordered
    state["errors"] = errors
    state["statistics"] = _stats(paragraphs, ordered, restored_count, controlled_occurrences, hard_controlled_occurrences, soft_controlled_occurrences, excluded_occurrences, placeholder_retries, placeholder_repaired, placeholder_fallback, placeholder_fallback_noncompliance, glossary_noncompliance, errors)
    if len(ordered) != len(paragraphs) or errors:
        state["status"] = "incomplete"
    elif state["statistics"].get("unresolved_paragraphs", 0) > 0:
        state["status"] = "completed_with_unresolved"
    else:
        state["status"] = "complete"
    state["updated_at"] = _now()
    atomic_write_json(output_path, state)
    return state


def _call_controlled_translation(
    *,
    client: Any,
    model: str,
    source_text: str,
    protected_source: str,
    constraints: list[dict[str, Any]],
    previous_context: str,
    constraint_mode: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    soft_constraints: list[dict[str, Any]] | None = None,
    placeholder_repair_policy: str = "retry_then_fallback",
    placeholder_quality_gate: bool = True,
    placeholder_semantic_verifier: str = "retry",
    placeholder_repeat_rejection_limit: int = 2,
    semantic_verifier_retries: int = 2,
) -> tuple[str, str, int, float, int, bool, dict[str, Any]]:
    """Translate one paragraph with validated HARD and SOFT occurrence controls.

    The default placeholder policy never edits a malformed Chinese translation merely
    to satisfy token counts. It performs fresh generations from the protected source,
    applies deterministic pre/post restoration gates, optionally verifies retried
    candidates semantically, and then falls back to the validated glossary.
    """
    started = time.monotonic()
    soft_constraints = soft_constraints or []
    last: Exception | None = None
    best_output = ""
    best_diagnostics: dict[str, Any] | None = None
    best_score = 10**9
    rejection_history: list[dict[str, Any]] = []
    model_request_count = 0
    semantic_verifier_calls = 0
    regeneration_attempts = 0
    legacy_repair_attempts = 0
    early_fallback_triggered = False
    contextual_omission_tokens: set[str] = set()
    max_tokens_escalations = 0
    repeat_rejection_counts: dict[str, int] = {}
    active_max_tokens = max_tokens

    def common_meta(**values: Any) -> dict[str, Any]:
        result = {
            "placeholder_strategy_version": PLACEHOLDER_STRATEGY_VERSION if constraint_mode == "placeholder" else None,
            "placeholder_repair_policy": placeholder_repair_policy if constraint_mode == "placeholder" else None,
            "placeholder_quality_gate_enabled": bool(placeholder_quality_gate) if constraint_mode == "placeholder" else None,
            "placeholder_semantic_verifier": placeholder_semantic_verifier if constraint_mode == "placeholder" else None,
            "placeholder_regeneration_attempts": regeneration_attempts,
            "placeholder_legacy_repair_attempts": legacy_repair_attempts,
            "placeholder_quality_gate_rejections": list(rejection_history),
            "placeholder_semantic_verifier_calls": semantic_verifier_calls,
            "placeholder_early_fallback": early_fallback_triggered,
            "placeholder_contextual_omission_tokens": sorted(contextual_omission_tokens),
            "placeholder_contextual_omission_count": len(contextual_omission_tokens),
            "definition_alias_normalizations": [],
            "definition_alias_normalized_occurrence_count": 0,
            "legitimate_definition_repetitions": [],
            "legitimate_definition_repetition_count": 0,
            "max_tokens_escalations": max_tokens_escalations,
            "model_request_count": model_request_count,
            "generation_retries": max(0, regeneration_attempts + legacy_repair_attempts),
            "glossary_compliance_retries": 0,
        }
        result.update(values)
        return result

    def register_rejection(stage: str, diagnostics: dict[str, Any], attempt_number: int) -> bool:
        """Persist a rejection and signal when an identical deterministic failure repeats.

        Re-generating the same malformed structure five times wastes calls and rarely
        changes the outcome. A zero limit disables early fallback.
        """
        nonlocal early_fallback_triggered
        rejection_history.append({
            "attempt": attempt_number,
            "stage": stage,
            "diagnostics": diagnostics,
        })
        signature = _quality_rejection_signature(stage, diagnostics)
        repeat_rejection_counts[signature] = repeat_rejection_counts.get(signature, 0) + 1
        if (
            placeholder_repeat_rejection_limit > 0
            and repeat_rejection_counts[signature] >= placeholder_repeat_rejection_limit
        ):
            early_fallback_triggered = True
            return True
        return False

    # A paragraph with no validated HARD occurrence must not be counted as a hard
    # placeholder success. It is translated with conditional hints only (or no controls).
    if not constraints:
        strategy = "conditional_glossary" if soft_constraints else "uncontrolled"
        for attempt in range(max_retries):
            try:
                model_request_count += 1
                output = _request_controlled_completion(
                    client=client,
                    model=model,
                    messages=_controlled_messages(
                        source_text=source_text,
                        protected_source=source_text,
                        constraints=[],
                        soft_constraints=soft_constraints,
                        previous_context=previous_context,
                        constraint_mode="glossary",
                        strict_reminder="",
                    ),
                    temperature=temperature,
                    max_tokens=active_max_tokens,
                )
                details = _glossary_compliance_details(output, [])
                return output, output, model_request_count, time.monotonic() - started, attempt, True, common_meta(
                    strategy=strategy,
                    placeholder_repair_attempts=0,
                    placeholder_fallback_used=False,
                    placeholder_diagnostics=None,
                    placeholder_inventory_diagnostics=None,
                    placeholder_context_diagnostics=None,
                    restored_translation_diagnostics=None,
                    placeholder_semantic_verifier_result=None,
                    glossary_compliance_details=details,
                    exact_glossary_compliance=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last = exc
                if isinstance(exc, ControlledTranslationTruncatedError):
                    increased = min(max(active_max_tokens * 2, active_max_tokens + 1024), 8192)
                    if increased > active_max_tokens:
                        active_max_tokens = increased
                        max_tokens_escalations += 1
                time.sleep(min(8.0, 1.5 ** attempt))
        assert last is not None
        raise last

    if constraint_mode == "glossary":
        final_output = ""
        final_details: dict[str, Any] = {}
        for attempt in range(max_retries):
            strict_reminder = (
                "" if attempt == 0 else
                "\n上一轮未完全遵守强制术语。请重新翻译，并逐一使用批准译法；条件性提示不得覆盖更长的其他法律名称。"
            )
            try:
                model_request_count += 1
                output = _request_controlled_completion(
                    client=client,
                    model=model,
                    messages=_controlled_messages(
                        source_text=source_text,
                        protected_source=protected_source,
                        constraints=constraints,
                        soft_constraints=soft_constraints,
                        previous_context=previous_context,
                        constraint_mode="glossary",
                        strict_reminder=strict_reminder,
                    ),
                    temperature=temperature,
                    max_tokens=active_max_tokens,
                )
                details = _glossary_compliance_details(output, constraints)
                final_output = output
                final_details = details
                if details["compliant"] or attempt == max_retries - 1:
                    return output, output, model_request_count, time.monotonic() - started, attempt, bool(details["compliant"]), common_meta(
                        strategy="glossary",
                        glossary_compliance_retries=attempt,
                        placeholder_repair_attempts=0,
                        placeholder_fallback_used=False,
                        placeholder_diagnostics=None,
                        placeholder_inventory_diagnostics=None,
                        placeholder_context_diagnostics=None,
                        restored_translation_diagnostics=None,
                        placeholder_semantic_verifier_result=None,
                        glossary_compliance_details=details,
                        exact_glossary_compliance=bool(details["exact_compliant"]),
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last = exc
                if isinstance(exc, ControlledTranslationTruncatedError):
                    increased = min(max(active_max_tokens * 2, active_max_tokens + 1024), 8192)
                    if increased > active_max_tokens:
                        active_max_tokens = increased
                        max_tokens_escalations += 1
            time.sleep(min(8.0, 1.5 ** attempt))
        if final_output:
            return final_output, final_output, model_request_count, time.monotonic() - started, max_retries - 1, bool(final_details.get("compliant")), common_meta(
                strategy="glossary",
                glossary_compliance_retries=max_retries - 1,
                placeholder_repair_attempts=0,
                placeholder_fallback_used=False,
                placeholder_diagnostics=None,
                placeholder_inventory_diagnostics=None,
                placeholder_context_diagnostics=None,
                restored_translation_diagnostics=None,
                placeholder_semantic_verifier_result=None,
                glossary_compliance_details=final_details,
                exact_glossary_compliance=bool(final_details.get("exact_compliant")),
            )
        assert last is not None
        raise last

    # Hard placeholder path. The safe default performs a fresh generation on every
    # attempt. Legacy LLM repair is retained only as an explicit ablation policy.
    for attempt in range(max_retries):
        inventory_diagnostics: dict[str, Any] | None = None
        context_diagnostics: dict[str, Any] | None = None
        restored_diagnostics: dict[str, Any] | None = None
        semantic_result: dict[str, Any] | None = None
        try:
            if attempt > 0 and placeholder_repair_policy == "llm_repair" and best_output:
                legacy_repair_attempts += 1
                messages = _placeholder_repair_messages(
                    protected_source=protected_source,
                    faulty_translation=best_output,
                    constraints=constraints,
                    diagnostics=best_diagnostics or _placeholder_diagnostics(best_output, constraints),
                )
            else:
                if attempt > 0:
                    regeneration_attempts += 1
                strict_reminder = "" if attempt == 0 else (
                    "\n上一轮输出未通过占位符或质量检查。请从受保护的英文原文重新完整翻译，"
                    "不要编辑、复制或延续上一轮中文。逐项保留每个占位符且各出现一次；"
                    "占位符旁不得再写其中文译法，不得为放置占位符而增加原文没有的内容。"
                )
                messages = _controlled_messages(
                    source_text=source_text,
                    protected_source=protected_source,
                    constraints=constraints,
                    soft_constraints=soft_constraints,
                    previous_context=previous_context,
                    constraint_mode="placeholder",
                    strict_reminder=strict_reminder,
                )

            model_request_count += 1
            output = _request_controlled_completion(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=active_max_tokens,
            )

            output, contextual_tokens, alias_diagnostics = _normalize_definitional_alias_wrappers(
                output,
                constraints,
            )
            active_constraints = [
                item for item in constraints
                if str(item.get("placeholder")) not in contextual_tokens
            ]

            inventory_diagnostics = _placeholder_diagnostics(
                output,
                constraints,
                contextually_satisfied=contextual_tokens,
            )
            inventory_diagnostics["definition_alias_normalizations"] = alias_diagnostics
            score = _placeholder_mismatch_score(inventory_diagnostics)
            if score < best_score:
                best_output = output
                best_diagnostics = inventory_diagnostics
                best_score = score
            if score != 0:
                last = PlaceholderMismatchError(inventory_diagnostics)
                if register_rejection("inventory", inventory_diagnostics, attempt + 1):
                    break
                time.sleep(min(8.0, 1.5 ** attempt))
                continue

            if placeholder_quality_gate:
                context_diagnostics = _placeholder_context_diagnostics(output, active_constraints)
                context_diagnostics["definition_alias_normalizations"] = alias_diagnostics
                if context_diagnostics["error_count"]:
                    last = PlaceholderQualityError("context", context_diagnostics)
                    if register_rejection("context", context_diagnostics, attempt + 1):
                        break
                    time.sleep(min(8.0, 1.5 ** attempt))
                    continue

            translation, restoration_spans = _restore_placeholders_with_spans(output, active_constraints)

            if placeholder_quality_gate:
                restored_diagnostics = _restored_translation_diagnostics(
                    translation,
                    active_constraints,
                    restoration_spans=restoration_spans,
                )
                restored_diagnostics["definition_alias_normalizations"] = alias_diagnostics
                if restored_diagnostics["error_count"]:
                    last = PlaceholderQualityError("restored", restored_diagnostics)
                    if register_rejection("restored", restored_diagnostics, attempt + 1):
                        break
                    time.sleep(min(8.0, 1.5 ** attempt))
                    continue

            should_verify = placeholder_semantic_verifier == "all" or (
                placeholder_semantic_verifier == "retry" and attempt > 0
            )
            if should_verify:
                model_request_count += 1
                semantic_verifier_calls += 1
                semantic_result = _verify_placeholder_translation_semantics(
                    client=client,
                    model=model,
                    source_text=source_text,
                    translation=translation,
                    constraints=constraints,
                    temperature=0.0,
                    max_tokens=min(max(active_max_tokens, 1600), 2400),
                    verifier_retries=semantic_verifier_retries,
                )
                if not semantic_result["passed"]:
                    last = PlaceholderQualityError("semantic", semantic_result)
                    if register_rejection("semantic", semantic_result, attempt + 1):
                        break
                    time.sleep(min(8.0, 1.5 ** attempt))
                    continue

            if attempt == 0:
                strategy = "placeholder_exact"
            elif placeholder_repair_policy == "llm_repair":
                strategy = "placeholder_repaired"
            else:
                strategy = "placeholder_retry_exact"
            contextual_omission_tokens.clear()
            contextual_omission_tokens.update(contextual_tokens)
            return translation, output, model_request_count, time.monotonic() - started, attempt, True, common_meta(
                strategy=strategy,
                placeholder_repair_attempts=legacy_repair_attempts,
                placeholder_fallback_used=False,
                placeholder_diagnostics=inventory_diagnostics,
                placeholder_inventory_diagnostics=inventory_diagnostics,
                placeholder_context_diagnostics=context_diagnostics,
                restored_translation_diagnostics=restored_diagnostics,
                placeholder_semantic_verifier_result=semantic_result,
                glossary_compliance_details=None,
                exact_glossary_compliance=None,
                definition_alias_normalizations=alias_diagnostics,
                definition_alias_normalized_occurrence_count=len(alias_diagnostics),
                legitimate_definition_repetitions=(restored_diagnostics or {}).get("legitimate_definition_repetitions", []),
                legitimate_definition_repetition_count=int((restored_diagnostics or {}).get("legitimate_definition_repetition_count", 0)),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            if not isinstance(exc, (PlaceholderMismatchError, PlaceholderQualityError)):
                diagnostics = {"error": f"{type(exc).__name__}: {exc}"}
                repeated = register_rejection("request_or_processing", diagnostics, attempt + 1)
                if isinstance(exc, ControlledTranslationTruncatedError):
                    increased = min(max(active_max_tokens * 2, active_max_tokens + 1024), 8192)
                    if increased > active_max_tokens:
                        active_max_tokens = increased
                        max_tokens_escalations += 1
                        repeated = False
                if repeated:
                    break
        time.sleep(min(8.0, 1.5 ** attempt))

    # Fall back to the validated mandatory glossary. SOFT occurrences remain
    # conditional hints and never count as compliance failures.
    fallback_last: Exception | None = None
    for fallback_attempt in range(max_retries):
        reminder = (
            "\n占位符生成未通过库存、上下文、恢复后或语义质量检查。"
            "现在不要输出任何占位符；请从原始英文完整翻译，并严格使用强制术语。"
        )
        try:
            model_request_count += 1
            output = _request_controlled_completion(
                client=client,
                model=model,
                messages=_controlled_messages(
                    source_text=source_text,
                    protected_source=protected_source,
                    constraints=constraints,
                    soft_constraints=soft_constraints,
                    previous_context=previous_context,
                    constraint_mode="glossary",
                    strict_reminder=reminder,
                ),
                temperature=temperature,
                max_tokens=active_max_tokens,
            )
            details = _glossary_compliance_details(output, constraints)
            compliance = bool(details["compliant"])
            if compliance or fallback_attempt == max_retries - 1:
                return output, output, model_request_count, time.monotonic() - started, max(0, max_retries - 1), compliance, common_meta(
                    strategy="glossary_fallback",
                    glossary_compliance_retries=fallback_attempt,
                    placeholder_repair_attempts=legacy_repair_attempts,
                    placeholder_fallback_used=True,
                    placeholder_diagnostics=best_diagnostics,
                    placeholder_inventory_diagnostics=best_diagnostics,
                    placeholder_context_diagnostics=None,
                    restored_translation_diagnostics=None,
                    placeholder_semantic_verifier_result=None,
                    placeholder_failure=f"{type(last).__name__}: {last}" if last else None,
                    glossary_compliance_details=details,
                    exact_glossary_compliance=bool(details["exact_compliant"]),
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            fallback_last = exc
            if isinstance(exc, ControlledTranslationTruncatedError):
                increased = min(max(active_max_tokens * 2, active_max_tokens + 1024), 8192)
                if increased > active_max_tokens:
                    active_max_tokens = increased
                    max_tokens_escalations += 1
            time.sleep(min(8.0, 1.5 ** fallback_attempt))
    if fallback_last is not None:
        raise fallback_last
    assert last is not None
    raise last

def _request_controlled_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise ControlledTranslationTruncatedError("controlled translation truncated by max_tokens")
    output = _clean_translation(choice.message.content)
    if not output:
        raise ValueError("model returned empty translation")
    return output


def _placeholder_repair_messages(
    *,
    protected_source: str,
    faulty_translation: str,
    constraints: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, str]]:
    required = [item["placeholder"] for item in constraints]
    ledger = []
    for item in constraints:
        ledger.append(
            f"{item['placeholder']} ← source occurrence {item.get('occurrence_index', 1)} "
            f"of {item.get('source_term') or item.get('term_id')}"
        )
    task = (
        "【受保护的英文原文】\n"
        f"{protected_source}\n\n"
        "【需要修复的中文译文】\n"
        f"{faulty_translation}\n\n"
        "【必须各出现一次的占位符清单】\n"
        f"{chr(10).join(ledger)}\n\n"
        "【当前问题】\n"
        f"missing={diagnostics.get('missing', [])}\n"
        f"duplicated={diagnostics.get('duplicated', {})}\n"
        f"unknown={diagnostics.get('unknown', {})}\n\n"
        "请返回修复后的完整中文译文。逐项核对下列 token，确保每个恰好出现一次：\n"
        + " ".join(required)
    )
    return [
        {"role": "system", "content": PLACEHOLDER_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def _controlled_messages(
    *,
    source_text: str,
    protected_source: str,
    constraints: list[dict[str, Any]],
    soft_constraints: list[dict[str, Any]],
    previous_context: str,
    constraint_mode: str,
    strict_reminder: str,
) -> list[dict[str, str]]:
    context = previous_context.strip() or "（无前文上下文）"
    mandatory_rows: list[str] = []
    conditional_rows: list[str] = []

    if constraint_mode == "placeholder":
        source = protected_source
        for item in constraints:
            mandatory_rows.append(
                f"{item['placeholder']}：对应定义术语 {item.get('source_term') or item.get('term_id')}；"
                f"批准译法为“{item['canonical_translation']}”；"
                f"法律范围摘要：{item.get('legal_scope_summary') or '以合同定义为准'}。"
                "输出中只保留占位符，不要同时输出批准译法。"
            )
        inventory = " ".join(item["placeholder"] for item in constraints) or "（无）"
        mandatory_policy = ("\n".join(mandatory_rows) or "（当前段落无硬约束术语）") + (
            "\n必须逐项保留且每项恰好出现一次的占位符清单：\n" + inventory
        )
        system = CONTROLLED_SYSTEM_PROMPT
    else:
        source = source_text
        seen: set[str] = set()
        for item in constraints:
            key = item["term_id"]
            if key in seen:
                continue
            seen.add(key)
            mandatory_rows.append(
                f"{item['source_term']} → {item['canonical_translation']}；"
                f"法律范围摘要：{item.get('legal_scope_summary') or '以合同定义为准'}。"
            )
        mandatory_policy = "\n".join(mandatory_rows) or "（当前段落无强制术语）"
        system = GLOSSARY_SYSTEM_PROMPT

    seen_soft: set[tuple[str, int, int]] = set()
    for item in soft_constraints:
        key = (str(item.get("term_id")), int(item.get("char_start", -1)), int(item.get("char_end", -1)))
        if key in seen_soft:
            continue
        seen_soft.add(key)
        reason = ", ".join(str(x) for x in item.get("validation_reasons", [])) or "上下文可能属于更长表达"
        compound = item.get("compound_context") or item.get("surface") or item.get("source_term")
        conditional_rows.append(
            f"{item.get('source_term')}（上下文：{compound}）候选译法为“{item.get('canonical_translation')}”。"
            f"仅当该处确实指本合同定义术语时采用；若它属于其他专名或普通表达，不得强制替换。判定依据：{reason}。"
        )
    conditional_policy = "\n".join(conditional_rows) or "（无条件性术语提示）"

    task = (
        "你正在逐段翻译同一份法律合同。\n"
        "【前文上下文】只用于理解语义，不得输出。\n\n"
        "================【强制术语控制】================\n"
        f"{mandatory_policy}\n"
        "================【强制术语控制结束】================\n\n"
        "================【条件性术语提示】================\n"
        f"{conditional_policy}\n"
        "================【条件性术语提示结束】================\n\n"
        "================【前文上下文：仅供参考】================\n"
        f"{context}\n"
        "================【前文上下文结束】================\n\n"
        "================【当前待译段落】================\n"
        f"{source}\n"
        "================【当前待译段落结束】================\n"
        f"{strict_reminder}\n"
        "只输出当前段落的完整中文译文。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def _all_term_renderings(item: dict[str, Any], *, include_risky: bool = True) -> list[dict[str, str]]:
    """Return known Chinese renderings with provenance for local contamination checks."""
    rows: list[dict[str, str]] = []

    def add(value: Any, kind: str) -> None:
        rendering = str(value or "").strip()
        if rendering and not any(x["rendering"] == rendering and x["kind"] == kind for x in rows):
            rows.append({"rendering": rendering, "kind": kind})

    add(item.get("canonical_translation"), "CANONICAL")
    for value in item.get("allowed_variants", []) or []:
        add(value.get("rendering") if isinstance(value, dict) else value, "ALLOWED_VARIANT")
    for value in item.get("subsequent_short_forms", []) or []:
        add(value.get("rendering") if isinstance(value, dict) else value, "SHORT_FORM")
    if include_risky:
        for value in item.get("risky_variants", []) or []:
            add(value.get("rendering") if isinstance(value, dict) else value, "RISKY_VARIANT")
    return rows


def _term_memory_collision_warnings(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report, but do not silently rewrite, cross-term canonical collisions."""
    by_rendering: dict[str, list[dict[str, str]]] = {}
    for item in terms:
        canonical = str(item.get("canonical_translation") or "").strip()
        if not canonical:
            continue
        by_rendering.setdefault(canonical, []).append({
            "term_id": str(item.get("term_id") or ""),
            "source_term": str(item.get("source_term") or ""),
        })
    warnings: list[dict[str, Any]] = []
    for canonical, rows in sorted(by_rendering.items()):
        unique = {(row["term_id"], row["source_term"]) for row in rows}
        source_terms = {row["source_term"] for row in rows if row["source_term"]}
        if len(unique) < 2 or len(source_terms) < 2:
            continue
        warnings.append({
            "issue_type": "CROSS_TERM_CANONICAL_COLLISION",
            "canonical_translation": canonical,
            "terms": rows,
            "message": "Different source terms share one canonical rendering; review legal distinguishability.",
        })
    return warnings


def _compact_for_adjacency(text: str) -> str:
    # Keep Chinese/Latin letters and digits while removing punctuation, quotes and spaces.
    return re.sub(r"[\s\u3000，。；：、,.!?！？（）()\[\]【】《》〈〉“”‘’\"'—–-]+", "", text)


def _quality_rejection_signature(stage: str, diagnostics: dict[str, Any]) -> str:
    """Return a stable signature for repeated deterministic failures."""
    if stage == "inventory":
        payload = {
            "stage": stage,
            "missing": sorted(diagnostics.get("missing") or []),
            "duplicated": sorted((diagnostics.get("duplicated") or {}).items()),
            "unknown": sorted((diagnostics.get("unknown") or {}).items()),
        }
    elif stage in {"context", "restored"}:
        payload = {
            "stage": stage,
            "issues": sorted(
                (
                    str(issue.get("issue_type")),
                    str(issue.get("term_id") or ""),
                    str(issue.get("placeholder") or ""),
                    str(issue.get("side") or ""),
                )
                for issue in (diagnostics.get("issues") or [])
                if issue.get("severity") == "ERROR"
            ),
        }
    elif stage == "semantic":
        payload = {
            "stage": stage,
            "adequacy": diagnostics.get("adequacy"),
            "term_semantics": diagnostics.get("term_semantics"),
            "unsupported_addition": diagnostics.get("unsupported_addition"),
            "meaning_omission": diagnostics.get("meaning_omission"),
            "unnatural_attachment": diagnostics.get("unnatural_attachment"),
            "broken_legal_logic": diagnostics.get("broken_legal_logic"),
            "problematic_terms": sorted(str(x) for x in diagnostics.get("problematic_terms") or []),
        }
    else:
        payload = {"stage": stage, "error": diagnostics.get("error")}
    return _json_hash(payload)


def _is_acronym_like(source_term: Any) -> bool:
    value = str(source_term or "").strip()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9&./-]{1,11}", value))


def _normalize_definitional_alias_wrappers(
    text: str,
    constraints: list[dict[str, Any]],
) -> tuple[str, set[str], list[dict[str, Any]]]:
    """Collapse redundant translated-full-form/placeholder alias wrappers.

    A legal definition can contain two target-side realizations of the same source
    occurrence, for example ``《联邦存款保险法》（“【术语0019】”）`` or
    ``【术语0004】（薪酬委员会）``. Restoring the placeholder would duplicate the
    Chinese term. This function removes only the redundant placeholder wrapper and
    records that occurrence as contextually satisfied.

    v1.7.2 applies the rule at term level rather than requiring the particular
    placeholder to be the DEFINIENDUM. Canonical renderings and approved variants
    are accepted; risky variants are not. Optional Chinese title marks and
    quoted/unquoted parentheses are preserved.
    """
    result = text
    satisfied: set[str] = set()
    events: list[dict[str, Any]] = []

    definiendum_term_ids = {
        str(item.get("term_id") or "")
        for item in constraints
        if str(item.get("usage_status") or "") == "DEFINIENDUM"
    }

    def approved_alias_renderings(item: dict[str, Any]) -> list[dict[str, str]]:
        rows = [
            row for row in _all_term_renderings(item)
            if row["kind"] in {"CANONICAL", "ALLOWED_VARIANT"}
        ]
        return sorted(rows, key=lambda row: len(row["rendering"]), reverse=True)

    def full_form_pattern(rendering: str) -> str:
        escaped = re.escape(rendering)
        if rendering.startswith("《") and rendering.endswith("》"):
            return rf"(?P<full>{escaped})"
        return rf"(?P<full>《\s*{escaped}\s*》|{escaped})"

    alias_word = r"(?:即|即为|简称|下称|以下简称|称为)?"
    quote = r"[‘’'\"“”]*"

    for item in constraints:
        token = str(item.get("placeholder") or "")
        term_id = str(item.get("term_id") or "")
        if not token or not term_id or term_id not in definiendum_term_ids:
            continue

        escaped_token = re.escape(token)
        normalized = False
        for rendering_row in approved_alias_renderings(item):
            rendering = str(rendering_row["rendering"] or "").strip()
            if not rendering:
                continue
            full = full_form_pattern(rendering)
            if rendering_row["kind"] == "CANONICAL":
                rendering_then_alias = "CANONICAL_THEN_ALIAS"
                alias_then_rendering = "ALIAS_THEN_CANONICAL"
            else:
                rendering_then_alias = "ALLOWED_VARIANT_THEN_ALIAS"
                alias_then_rendering = "ALIAS_THEN_ALLOWED_VARIANT"
            patterns: list[tuple[str, str]] = [
                (
                    rf"{full}\s*[（(]\s*{alias_word}\s*{quote}\s*{escaped_token}\s*{quote}\s*[）)]",
                    rendering_then_alias,
                ),
                (
                    rf"{escaped_token}\s*[（(]\s*{alias_word}\s*{quote}\s*{full}\s*{quote}\s*[）)]",
                    alias_then_rendering,
                ),
            ]
            for pattern, pattern_type in patterns:
                match = re.search(pattern, result)
                if not match:
                    continue
                matched_full = str(match.group("full"))
                before = result[max(0, match.start() - 24):match.end() + 24]
                result = result[:match.start()] + matched_full + result[match.end():]
                satisfied.add(token)
                events.append({
                    "placeholder": token,
                    "term_id": item.get("term_id"),
                    "source_term": item.get("source_term"),
                    "canonical_translation": item.get("canonical_translation"),
                    "matched_rendering": rendering,
                    "rendering_kind": rendering_row["kind"],
                    "pattern": pattern_type,
                    "original_context": before,
                    "replacement": matched_full,
                })
                normalized = True
                break
            if normalized:
                break

    satisfied_term_ids = {
        str(event.get("term_id")) for event in events if event.get("term_id") is not None
    }
    for item in constraints:
        token = str(item.get("placeholder") or "")
        term_id = str(item.get("term_id") or "")
        if not token or token in satisfied or term_id not in satisfied_term_ids:
            continue

        normalized = False
        for rendering_row in approved_alias_renderings(item):
            rendering = str(rendering_row["rendering"] or "").strip()
            if not rendering:
                continue
            escaped_token = re.escape(token)
            full = full_form_pattern(rendering)
            if rendering_row["kind"] == "CANONICAL":
                sibling_before = "SIBLING_ALIAS_BEFORE_CANONICAL"
                sibling_after = "SIBLING_ALIAS_AFTER_CANONICAL"
            else:
                sibling_before = "SIBLING_ALIAS_BEFORE_ALLOWED_VARIANT"
                sibling_after = "SIBLING_ALIAS_AFTER_ALLOWED_VARIANT"
            for pattern, pattern_type in [
                (rf"{escaped_token}\s*{full}", sibling_before),
                (rf"{full}\s*{escaped_token}", sibling_after),
            ]:
                match = re.search(pattern, result)
                if not match:
                    continue
                matched_full = str(match.group("full"))
                before = result[max(0, match.start() - 24):match.end() + 24]
                result = result[:match.start()] + matched_full + result[match.end():]
                satisfied.add(token)
                events.append({
                    "placeholder": token,
                    "term_id": item.get("term_id"),
                    "source_term": item.get("source_term"),
                    "canonical_translation": item.get("canonical_translation"),
                    "matched_rendering": rendering,
                    "rendering_kind": rendering_row["kind"],
                    "pattern": pattern_type,
                    "original_context": before,
                    "replacement": matched_full,
                })
                normalized = True
                break
            if normalized:
                break

    # A model may translate a title-marked legal instrument and then retain the
    # placeholder immediately beside it, for example::
    #
    #     《国内税收法典》【术语0006_01】第280G条
    #
    # Restoring the placeholder would yield a duplicate title.  This is not the
    # same as ``《【术语0006_01】》``: the latter still needs the placeholder and is
    # rejected later as a redundant title-mark wrapper.  Only a complete approved
    # rendering that already includes its own Chinese title marks is eligible for
    # this deterministic normalization.
    for item in constraints:
        token = str(item.get("placeholder") or "")
        if not token or token in satisfied:
            continue

        normalized = False
        for rendering_row in approved_alias_renderings(item):
            rendering = str(rendering_row["rendering"] or "").strip()
            if not (rendering.startswith("《") and rendering.endswith("》")):
                continue

            escaped_token = re.escape(token)
            escaped_rendering = re.escape(rendering)
            if rendering_row["kind"] == "CANONICAL":
                before_type = "TITLE_MARKED_CANONICAL_BEFORE_PLACEHOLDER"
                after_type = "PLACEHOLDER_BEFORE_TITLE_MARKED_CANONICAL"
            else:
                before_type = "TITLE_MARKED_ALLOWED_VARIANT_BEFORE_PLACEHOLDER"
                after_type = "PLACEHOLDER_BEFORE_TITLE_MARKED_ALLOWED_VARIANT"

            patterns = [
                (
                    rf"(?P<full>{escaped_rendering})\s*{escaped_token}",
                    before_type,
                ),
                (
                    rf"{escaped_token}\s*(?P<full>{escaped_rendering})",
                    after_type,
                ),
            ]
            for pattern, pattern_type in patterns:
                match = re.search(pattern, result)
                if not match:
                    continue

                matched_full = str(match.group("full"))
                before = result[max(0, match.start() - 24):match.end() + 24]
                result = result[:match.start()] + matched_full + result[match.end():]
                satisfied.add(token)
                events.append({
                    "placeholder": token,
                    "term_id": item.get("term_id"),
                    "source_term": item.get("source_term"),
                    "canonical_translation": item.get("canonical_translation"),
                    "matched_rendering": rendering,
                    "rendering_kind": rendering_row["kind"],
                    "pattern": pattern_type,
                    "original_context": before,
                    "replacement": matched_full,
                })
                normalized = True
                break
            if normalized:
                break

    return result, satisfied, events



def _placeholder_context_diagnostics(
    text: str,
    constraints: list[dict[str, Any]],
    *,
    window_chars: int = 28,
) -> dict[str, Any]:
    """Detect approved/risky Chinese wording already written beside placeholders.

    The model must emit the placeholder alone for the protected source occurrence.
    Writing a Chinese rendering next to it causes double translation after restoration.
    """
    issues: list[dict[str, Any]] = []
    for item in constraints:
        token = str(item["placeholder"])
        positions = [match.start() for match in re.finditer(re.escape(token), text)]
        if len(positions) != 1:
            continue
        pos = positions[0]
        left = text[max(0, pos - window_chars):pos]
        right = text[pos + len(token):pos + len(token) + window_chars]
        left_compact = _compact_for_adjacency(left)
        right_compact = _compact_for_adjacency(right)
        for rendering in _all_term_renderings(item):
            value = _compact_for_adjacency(rendering["rendering"])
            if not value:
                continue
            side = None
            if left_compact.endswith(value):
                side = "LEFT"
            elif right_compact.startswith(value):
                side = "RIGHT"
            if side:
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "PLACEHOLDER_WRAPPER_CONTAMINATION",
                    "placeholder": token,
                    "term_id": item.get("term_id"),
                    "source_term": item.get("source_term"),
                    "rendering": rendering["rendering"],
                    "rendering_kind": rendering["kind"],
                    "side": side,
                    "local_context": text[max(0, pos - window_chars):pos + len(token) + window_chars],
                })
                break

        canonical = str(item.get("canonical_translation", ""))
        if canonical.startswith("《") and canonical.endswith("》"):
            stripped_left = left.rstrip()
            stripped_right = right.lstrip()
            if stripped_left.endswith("《") and stripped_right.startswith("》"):
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "REDUNDANT_TITLE_MARK_WRAPPER",
                    "placeholder": token,
                    "term_id": item.get("term_id"),
                    "local_context": text[max(0, pos - window_chars):pos + len(token) + window_chars],
                })

    return {
        "issues": issues,
        "error_count": sum(1 for x in issues if x.get("severity") == "ERROR"),
        "warning_count": sum(1 for x in issues if x.get("severity") == "WARNING"),
    }


def _restored_translation_diagnostics(
    text: str,
    constraints: list[dict[str, Any]],
    *,
    restoration_spans: list[dict[str, Any]] | None = None,
    window_chars: int = 28,
) -> dict[str, Any]:
    """Validate restored output using restoration-span accounting.

    Every canonical occurrence inserted by restoration is accounted for by an exact
    target span. Multiple accounted occurrences of the same legal term are legitimate,
    including heading-plus-definiendum and definition-then-use patterns. Only an
    unaccounted copy directly adjacent to an accounted span is rejected.
    """
    issues: list[dict[str, Any]] = []
    legitimate_repetitions: list[dict[str, Any]] = []
    if PLACEHOLDER_RE.search(text):
        issues.append({
            "severity": "ERROR",
            "issue_type": "UNRESTORED_PLACEHOLDER",
            "local_context": PLACEHOLDER_RE.search(text).group(0),
        })

    item_by_placeholder = {
        str(item.get("placeholder")): item for item in constraints if item.get("placeholder")
    }
    if restoration_spans is None:
        spans: list[dict[str, Any]] = []
        seen_probe: set[tuple[str, int, int]] = set()
        for item in constraints:
            canonical = str(item.get("canonical_translation") or "").strip()
            if not canonical:
                continue
            for match in re.finditer(re.escape(canonical), text):
                key = (str(item.get("term_id") or ""), match.start(), match.end())
                if key in seen_probe:
                    continue
                seen_probe.add(key)
                spans.append({
                    "placeholder": item.get("placeholder"),
                    "term_id": item.get("term_id"),
                    "source_term": item.get("source_term"),
                    "canonical_translation": canonical,
                    "start": match.start(),
                    "end": match.end(),
                })
    else:
        spans = list(restoration_spans)

    accounted_by_key: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for span in spans:
        term_id = str(span.get("term_id") or "")
        canonical = str(span.get("canonical_translation") or "").strip()
        if canonical:
            accounted_by_key.setdefault((term_id, canonical), []).append(
                (int(span.get("start", 0)), int(span.get("end", 0)))
            )

    definition_cue_re = re.compile(
        r"(?:shall\s+mean|means?|是指|指的是|应指|含义|以下简称|下称)", re.I
    )
    grouped_spans: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for span in spans:
        key = (
            str(span.get("term_id") or ""),
            str(span.get("canonical_translation") or "").strip(),
        )
        if key[1]:
            grouped_spans.setdefault(key, []).append(span)
    for (term_id, canonical), rows in grouped_spans.items():
        rows = sorted(rows, key=lambda row: int(row.get("start", 0)))
        for left_span, right_span in zip(rows, rows[1:]):
            gap_start = int(left_span.get("end", 0))
            gap_end = int(right_span.get("start", 0))
            if gap_end < gap_start or gap_end - gap_start > max(96, window_chars * 4):
                continue
            between = text[gap_start:gap_end]
            local = text[
                max(0, int(left_span.get("start", 0)) - 12):
                min(len(text), int(right_span.get("end", 0)) + 40)
            ]
            if definition_cue_re.search(local) or re.search(
                r"[。.;；：:]\s*[‘’'\"“”]", between
            ):
                legitimate_repetitions.append({
                    "term_id": term_id,
                    "canonical_translation": canonical,
                    "left_placeholder": left_span.get("placeholder"),
                    "right_placeholder": right_span.get("placeholder"),
                    "local_context": local,
                })

    for span in spans:
        item = item_by_placeholder.get(str(span.get("placeholder"))) or span
        term_id = str(item.get("term_id") or "")
        canonical = str(
            span.get("canonical_translation") or item.get("canonical_translation") or ""
        ).strip()
        if not canonical:
            continue
        start = int(span.get("start", 0))
        end = int(span.get("end", start + len(canonical)))
        left = text[max(0, start - window_chars):start]
        right = text[end:end + window_chars]
        local_context = text[
            max(0, start - window_chars):min(len(text), end + window_chars)
        ]

        all_matches = [(m.start(), m.end()) for m in re.finditer(re.escape(canonical), text)]
        accounted = accounted_by_key.get((term_id, canonical), [])
        unaccounted = [
            match_span for match_span in all_matches
            if not any(
                match_span[0] < acc_end and acc_start < match_span[1]
                for acc_start, acc_end in accounted
            )
        ]
        for extra_start, extra_end in unaccounted:
            if extra_end <= start:
                between = text[extra_end:start]
            elif end <= extra_start:
                between = text[end:extra_start]
            else:
                continue
            if not re.fullmatch(r"\s*", between):
                continue
            issues.append({
                "severity": "ERROR",
                "issue_type": "DUPLICATED_CANONICAL_RENDERING",
                "term_id": term_id,
                "source_term": item.get("source_term"),
                "canonical_translation": canonical,
                "placeholder": span.get("placeholder"),
                "unaccounted_span": {"start": extra_start, "end": extra_end},
                "local_context": text[
                    max(0, min(start, extra_start) - window_chars):
                    min(len(text), max(end, extra_end) + window_chars)
                ],
            })
            break

        for other_start, other_end in accounted:
            if (other_start, other_end) == (start, end):
                continue
            if other_end <= start:
                between = text[other_end:start]
            elif end <= other_start:
                between = text[end:other_start]
            else:
                continue
            if re.fullmatch(r"\s*", between):
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "ADJACENT_RESTORED_CANONICAL_DUPLICATION",
                    "term_id": term_id,
                    "source_term": item.get("source_term"),
                    "canonical_translation": canonical,
                    "placeholder": span.get("placeholder"),
                    "local_context": text[
                        max(0, min(start, other_start) - window_chars):
                        min(len(text), max(end, other_end) + window_chars)
                    ],
                })
                break

        if canonical.startswith("《") and canonical.endswith("》"):
            if left.rstrip().endswith("《") or right.lstrip().startswith("》"):
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "NESTED_TITLE_MARKS",
                    "term_id": term_id,
                    "placeholder": span.get("placeholder"),
                    "local_context": local_context,
                })

        compact_canonical = _compact_for_adjacency(canonical)
        left_compact = _compact_for_adjacency(left)
        right_compact = _compact_for_adjacency(right)
        for rendering in _all_term_renderings(item, include_risky=True):
            if rendering["kind"] != "RISKY_VARIANT":
                continue
            risky = _compact_for_adjacency(rendering["rendering"])
            if not risky or not compact_canonical or risky == compact_canonical:
                continue
            if left_compact.endswith(risky) or right_compact.startswith(risky):
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "RISKY_AND_CANONICAL_ADJACENCY",
                    "term_id": term_id,
                    "source_term": item.get("source_term"),
                    "risky_rendering": rendering["rendering"],
                    "canonical_translation": canonical,
                    "placeholder": span.get("placeholder"),
                    "local_context": local_context,
                })

        if canonical.endswith("期") and right.startswith("期间"):
            issues.append({
                "severity": "WARNING",
                "issue_type": "POSSIBLE_PERIOD_SUFFIX_REDUNDANCY",
                "term_id": term_id,
                "text": canonical + "期间",
                "local_context": local_context,
            })

    deduped: list[dict[str, Any]] = []
    seen_issue_keys: set[str] = set()
    for issue in issues:
        key = json.dumps({
            "issue_type": issue.get("issue_type"),
            "term_id": issue.get("term_id"),
            "placeholder": issue.get("placeholder"),
            "local_context": issue.get("local_context"),
        }, ensure_ascii=False, sort_keys=True)
        if key in seen_issue_keys:
            continue
        seen_issue_keys.add(key)
        deduped.append(issue)

    return {
        "issues": deduped,
        "error_count": sum(1 for x in deduped if x.get("severity") == "ERROR"),
        "warning_count": sum(1 for x in deduped if x.get("severity") == "WARNING"),
        "legitimate_definition_repetitions": legitimate_repetitions,
        "legitimate_definition_repetition_count": len(legitimate_repetitions),
        "restoration_span_count": len(spans),
    }



def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(cleaned[start:end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("semantic verifier did not return a JSON object")


def _verify_placeholder_translation_semantics(
    *,
    client: Any,
    model: str,
    source_text: str,
    translation: str,
    constraints: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    verifier_retries: int = 2,
) -> dict[str, Any]:
    unique_terms: list[str] = []
    seen: set[str] = set()
    for item in constraints:
        term_id = str(item.get("term_id"))
        if term_id in seen:
            continue
        seen.add(term_id)
        unique_terms.append(
            f"{item.get('source_term')} → {item.get('canonical_translation')}"
        )
    task = (
        "【英文原文】\n" + source_text + "\n\n"
        "【候选中文译文】\n" + translation + "\n\n"
        "【受保护术语】\n" + "\n".join(unique_terms) + "\n\n"
        "请严格核验，并只返回指定 JSON。"
    )
    last: Exception | None = None
    raw = ""
    data: dict[str, Any] | None = None
    for verifier_attempt in range(verifier_retries):
        try:
            raw = _request_controlled_completion(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": SEMANTIC_VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            data = _extract_json_object(raw)
            break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            if verifier_attempt + 1 < verifier_retries:
                time.sleep(min(2.0, 0.5 * (verifier_attempt + 1)))
    if data is None:
        assert last is not None
        raise last
    required_bools = [
        "unsupported_addition",
        "meaning_omission",
        "unnatural_attachment",
        "broken_legal_logic",
    ]
    normalized = {
        "adequacy": str(data.get("adequacy", "FAIL")).upper(),
        "term_semantics": str(data.get("term_semantics", "FAIL")).upper(),
        **{key: bool(data.get(key, True)) for key in required_bools},
        "problematic_terms": list(data.get("problematic_terms") or []),
        "rationale": str(data.get("rationale", "")),
        "raw": raw,
    }
    normalized["passed"] = (
        normalized["adequacy"] == "PASS"
        and normalized["term_semantics"] == "PASS"
        and not any(normalized[key] for key in required_bools)
    )
    return normalized

def _placeholder_diagnostics(
    text: str,
    constraints: list[dict[str, Any]],
    *,
    contextually_satisfied: set[str] | None = None,
) -> dict[str, Any]:
    contextually_satisfied = contextually_satisfied or set()
    expected_tokens = [item["placeholder"] for item in constraints]
    expected = {token: 1 for token in expected_tokens}
    observed: dict[str, int] = {}
    for token in PLACEHOLDER_RE.findall(text):
        observed[token] = observed.get(token, 0) + 1
    missing = [
        token for token in expected_tokens
        if observed.get(token, 0) == 0 and token not in contextually_satisfied
    ]
    duplicated = {token: count for token, count in observed.items() if token in expected and count > 1}
    unknown = {token: count for token, count in observed.items() if token not in expected}
    return {
        "expected": expected,
        "observed": observed,
        "missing": missing,
        "duplicated": duplicated,
        "unknown": unknown,
        "contextually_satisfied": sorted(contextually_satisfied),
    }


def _placeholder_mismatch_score(diagnostics: dict[str, Any]) -> int:
    duplicate_excess = sum(max(0, int(count) - 1) for count in diagnostics.get("duplicated", {}).values())
    unknown_count = sum(int(count) for count in diagnostics.get("unknown", {}).values())
    return len(diagnostics.get("missing", [])) + duplicate_excess + unknown_count


def _validate_placeholder_output(text: str, constraints: list[dict[str, Any]]) -> None:
    diagnostics = _placeholder_diagnostics(text, constraints)
    if _placeholder_mismatch_score(diagnostics) != 0:
        raise PlaceholderMismatchError(diagnostics)


def _restore_placeholders_with_spans(
    text: str,
    constraints: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    item_by_placeholder: dict[str, dict[str, Any]] = {}
    for item in constraints:
        placeholder = item["placeholder"]
        value = item["canonical_translation"]
        if placeholder in mapping and mapping[placeholder] != value:
            raise ValueError(f"conflicting canonical translations for {placeholder}")
        mapping[placeholder] = value
        item_by_placeholder[placeholder] = item

    chunks: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    output_length = 0
    for match in PLACEHOLDER_RE.finditer(text):
        token = match.group(0)
        if token not in mapping:
            continue
        prefix = text[cursor:match.start()]
        chunks.append(prefix)
        output_length += len(prefix)
        value = mapping[token]
        start = output_length
        chunks.append(value)
        output_length += len(value)
        item = item_by_placeholder[token]
        spans.append({
            "placeholder": token,
            "term_id": item.get("term_id"),
            "source_term": item.get("source_term"),
            "canonical_translation": value,
            "start": start,
            "end": output_length,
        })
        cursor = match.end()
    chunks.append(text[cursor:])
    result = "".join(chunks)
    if PLACEHOLDER_RE.search(result):
        raise ValueError("unrestored placeholder remains in final translation")
    return result, spans


def _restore_placeholders(text: str, constraints: list[dict[str, Any]]) -> str:
    result, _ = _restore_placeholders_with_spans(text, constraints)
    return result


def _glossary_compliance_details(text: str, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    by_term: dict[str, dict[str, Any]] = {}
    for item in constraints:
        by_term.setdefault(str(item.get("term_id")), item)

    rows: list[dict[str, Any]] = []
    for term_id, item in sorted(by_term.items()):
        canonical = str(item.get("canonical_translation", "")).strip()
        approved = _approved_renderings(item)
        exact = bool(canonical and canonical in text)
        matched = canonical if exact else next((value for value in approved if value and value in text), None)
        status = "EXACT_CANONICAL" if exact else ("APPROVED_VARIANT" if matched else "MISSING_REQUIRED_RENDERING")
        rows.append({
            "term_id": term_id,
            "source_term": item.get("source_term"),
            "canonical_translation": canonical,
            "approved_renderings": approved,
            "status": status,
            "matched_rendering": matched,
        })
    return {
        "compliant": all(row["status"] != "MISSING_REQUIRED_RENDERING" for row in rows),
        "exact_compliant": all(row["status"] == "EXACT_CANONICAL" for row in rows),
        "terms": rows,
        "missing_term_ids": [row["term_id"] for row in rows if row["status"] == "MISSING_REQUIRED_RENDERING"],
        "approved_variant_term_ids": [row["term_id"] for row in rows if row["status"] == "APPROVED_VARIANT"],
    }


def _glossary_compliance(text: str, constraints: list[dict[str, Any]]) -> bool:
    return bool(_glossary_compliance_details(text, constraints)["compliant"])


def _approved_renderings(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    canonical = str(item.get("canonical_translation", "")).strip()
    if canonical:
        values.append(canonical)

    informal = re.compile(r"非正式|口语|内部沟通|内部文件|informal|oral", re.I)
    for variant in item.get("allowed_variants", []) or []:
        if isinstance(variant, str):
            rendering = variant.strip()
            condition = ""
        elif isinstance(variant, dict):
            rendering = str(variant.get("rendering", "")).strip()
            condition = str(variant.get("condition", ""))
        else:
            continue
        if rendering and not informal.search(condition):
            values.append(rendering)

    for value in item.get("subsequent_short_forms", []) or []:
        if isinstance(value, dict):
            rendering = str(value.get("rendering", "")).strip()
        else:
            rendering = str(value).strip()
        if rendering:
            values.append(rendering)

    return list(dict.fromkeys(values))


def _placeholder_for_term(term_id: str) -> str:
    digits = "".join(ch for ch in term_id if ch.isdigit())
    if not digits:
        digits = str(int(hashlib.sha256(term_id.encode("utf-8")).hexdigest()[:8], 16))
    return f"【术语{digits.zfill(4)}】"


def _load_memories(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.term-memory.json"))
    values: dict[str, dict[str, Any]] = {}
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        document_id = str(data.get("document_id") or "")
        if document_id:
            values[document_id] = data
    return values


def _iter_documents(path: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "paragraphs" in data:
            yield path, data
        return
    for file in sorted(path.rglob("*.json")):
        if file.name.endswith("manifest.json"):
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "paragraphs" in data:
            yield file, data


def _stats(
    paragraphs: list[dict[str, Any]],
    translations: list[dict[str, Any]],
    restored: int,
    controlled_occurrences: int,
    hard_controlled_occurrences: int,
    soft_controlled_occurrences: int,
    excluded_occurrences: int,
    placeholder_retries: int,
    placeholder_repaired: int,
    placeholder_fallback: int,
    placeholder_fallback_noncompliance: int,
    glossary_noncompliance: int,
    errors: list[dict[str, Any]],
) -> dict[str, int]:
    # Derive audit totals from persisted rows so resumed runs report the full
    # experiment rather than only work completed in the current process.
    controlled_occurrences = sum(int(x.get("controlled_term_occurrence_count", 0)) for x in translations)
    hard_controlled_occurrences = sum(int(x.get("hard_controlled_term_occurrence_count", 0)) for x in translations)
    soft_controlled_occurrences = sum(int(x.get("soft_controlled_term_occurrence_count", 0)) for x in translations)
    excluded_occurrences = sum(int(x.get("excluded_term_occurrence_count", 0)) for x in translations)
    generation_retries = sum(int(x.get("generation_retry_count", x.get("placeholder_retry_count", 0))) for x in translations)
    placeholder_retries = generation_retries  # backward-compatible alias
    glossary_compliance_retries = sum(int(x.get("glossary_compliance_retry_count", 0)) for x in translations)
    placeholder_exact = sum(1 for x in translations if x.get("constraint_strategy_used") == "placeholder_exact")
    placeholder_retry_success = sum(1 for x in translations if x.get("constraint_strategy_used") == "placeholder_retry_exact")
    placeholder_repaired = sum(1 for x in translations if x.get("constraint_strategy_used") == "placeholder_repaired")
    conditional_glossary = sum(1 for x in translations if x.get("constraint_strategy_used") == "conditional_glossary")
    uncontrolled = sum(1 for x in translations if x.get("constraint_strategy_used") == "uncontrolled")
    placeholder_fallback = sum(1 for x in translations if x.get("placeholder_fallback_used"))
    placeholder_fallback_noncompliance = sum(
        1 for x in translations if x.get("placeholder_fallback_used") and not x.get("constraint_compliance", False)
    )
    placeholder_fallback_verified = sum(
        1 for row in translations if row.get("resolution_status") == "glossary_fallback_verified"
    )
    unresolved_paragraphs = sum(
        1 for row in translations if row.get("resolution_status") in { "glossary_fallback_noncompliant", "noncompliant",}
    )
    glossary_noncompliance = sum(
        1 for x in translations
        if x.get("constraint_mode") == "glossary" and not x.get("constraint_compliance", False)
    )
    regeneration_attempts = sum(int(x.get("placeholder_regeneration_attempts", 0)) for x in translations)
    legacy_repair_attempts = sum(int(x.get("placeholder_legacy_repair_attempts", 0)) for x in translations)
    semantic_calls = sum(int(x.get("placeholder_semantic_verifier_calls", 0)) for x in translations)
    early_fallbacks = sum(1 for x in translations if x.get("placeholder_early_fallback"))
    contextual_omission_paragraphs = sum(
        1 for x in translations if int(x.get("placeholder_contextual_omission_count", 0)) > 0
    )
    contextual_omission_occurrences = sum(
        int(x.get("placeholder_contextual_omission_count", 0)) for x in translations
    )
    definition_alias_normalized_paragraphs = sum(
        1 for x in translations if int(x.get("definition_alias_normalized_occurrence_count", 0)) > 0
    )
    definition_alias_normalized_occurrences = sum(
        int(x.get("definition_alias_normalized_occurrence_count", 0)) for x in translations
    )
    legitimate_definition_repetition_paragraphs = sum(
        1 for x in translations if int(x.get("legitimate_definition_repetition_count", 0)) > 0
    )
    legitimate_definition_repetition_pairs = sum(
        int(x.get("legitimate_definition_repetition_count", 0)) for x in translations
    )
    max_tokens_escalations = sum(int(x.get("max_tokens_escalations", 0)) for x in translations)
    quality_rejections = [
        rejection
        for row in translations
        for rejection in (row.get("placeholder_quality_gate_rejections") or [])
    ]
    return {
        "total_paragraphs": len(paragraphs),
        "translated_paragraphs": len(translations),
        "restored_paragraphs": restored,
        "failed_paragraphs": len({str(x.get('paragraph_id')) for x in errors}),
        "remaining_paragraphs": max(0, len(paragraphs) - len(translations)),
        "controlled_term_occurrences": controlled_occurrences,
        "hard_controlled_term_occurrences": hard_controlled_occurrences,
        "soft_controlled_term_occurrences": soft_controlled_occurrences,
        "excluded_term_occurrences": excluded_occurrences,
        "placeholder_retries": placeholder_retries,
        "generation_retries": generation_retries,
        "glossary_compliance_retries": glossary_compliance_retries,
        "placeholder_regeneration_attempts": regeneration_attempts,
        "placeholder_legacy_repair_attempts": legacy_repair_attempts,
        "placeholder_quality_gate_rejections": len(quality_rejections),
        "placeholder_inventory_rejections": sum(1 for x in quality_rejections if x.get("stage") == "inventory"),
        "placeholder_context_rejections": sum(1 for x in quality_rejections if x.get("stage") == "context"),
        "placeholder_restored_rejections": sum(1 for x in quality_rejections if x.get("stage") == "restored"),
        "placeholder_semantic_verifier_calls": semantic_calls,
        "placeholder_semantic_rejections": sum(1 for x in quality_rejections if x.get("stage") == "semantic"),
        "placeholder_request_or_processing_rejections": sum(
            1 for x in quality_rejections if x.get("stage") == "request_or_processing"
        ),
        "placeholder_early_fallback_paragraphs": early_fallbacks,
        "placeholder_contextual_omission_paragraphs": contextual_omission_paragraphs,
        "placeholder_contextual_omission_occurrences": contextual_omission_occurrences,
        "definition_alias_normalized_paragraphs": definition_alias_normalized_paragraphs,
        "definition_alias_normalized_occurrences": definition_alias_normalized_occurrences,
        "legitimate_definition_repetition_paragraphs": legitimate_definition_repetition_paragraphs,
        "legitimate_definition_repetition_pairs": legitimate_definition_repetition_pairs,
        "max_tokens_escalations": max_tokens_escalations,
        "placeholder_exact_paragraphs": placeholder_exact,
        "placeholder_retry_success_paragraphs": placeholder_retry_success,
        "placeholder_repaired_paragraphs": placeholder_repaired,
        "conditional_glossary_paragraphs": conditional_glossary,
        "uncontrolled_paragraphs": uncontrolled,
        "placeholder_fallback_paragraphs": placeholder_fallback,
        "placeholder_fallback_noncompliance_paragraphs": placeholder_fallback_noncompliance,
        "glossary_noncompliance_paragraphs": glossary_noncompliance,
        "placeholder_fallback_verified_paragraphs": placeholder_fallback_verified,
        "unresolved_paragraphs": unresolved_paragraphs,
    }

def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
