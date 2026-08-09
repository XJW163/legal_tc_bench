from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import httpx
from tqdm import tqdm

from .io_utils import atomic_write_json, exclusive_process_lock


EVALUATION_SCHEMA_VERSION = "1.1"
JUDGE_PROMPT_VERSION = "blind-pairwise-legal-translation-v2-low-cost"
EVALUATION_SCOPES = {"all", "term-bearing", "term-bearing-plus-sample"}
SCORE_FIELDS = (
    "semantic_fidelity",
    "legal_accuracy",
    "terminology_compliance",
    "fluency",
    "overall_quality",
)
CRITICAL_ERROR_FIELDS = (
    "meaning_omission",
    "unsupported_addition",
    "polarity_or_modality_error",
    "party_or_role_error",
    "number_date_or_reference_error",
)


class TranslationEvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, int] | None = None,
        last_raw: str = "",
    ) -> None:
        super().__init__(message)
        self.usage = usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_raw = last_raw


def evaluate_translation_tree(
    annotated: Path,
    baseline: Path,
    controlled: Path,
    output: Path,
    *,
    experiment_name: str,
    model: str,
    base_url: str,
    term_memory: Path | None = None,
    api_key_env: str | None = None,
    baseline_label: str = "baseline",
    controlled_label: str = "controlled",
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_retries: int = 2,
    timeout_seconds: float = 300.0,
    workers: int = 8,
    position_control: str = "deterministic",
    save_raw: bool = False,
    overwrite: bool = False,
    max_documents: int | None = None,
    document_ids: set[str] | None = None,
    max_paragraphs_per_document: int | None = None,
    extra_body: dict[str, Any] | None = None,
    evaluation_scope: str = "term-bearing-plus-sample",
    non_term_sample_rate: float = 0.1,
    sample_seed: int = 2026,
    json_mode: bool = True,
    max_token_escalations: int = 1,
) -> dict[str, Any]:
    """Blindly compare baseline and controlled translations with an LLM judge.

    A single paragraph or document failure is recorded and never aborts the whole run.
    Existing paragraph judgments are restored when the judge settings and paragraph
    inputs are unchanged.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if position_control not in {"fixed", "deterministic", "bidirectional"}:
        raise ValueError(
            "position_control must be one of: fixed, deterministic, bidirectional"
        )
    if evaluation_scope not in EVALUATION_SCOPES:
        raise ValueError(
            "evaluation_scope must be one of: all, term-bearing, "
            "term-bearing-plus-sample"
        )
    if not 0.0 <= non_term_sample_rate <= 1.0:
        raise ValueError("non_term_sample_rate must be between 0 and 1")
    if max_token_escalations < 0:
        raise ValueError("max_token_escalations must be >= 0")

    annotated_index = _index_documents(annotated, required_key="paragraphs")
    baseline_index = _index_documents(baseline, required_key="translations")
    controlled_index = _index_documents(controlled, required_key="translations")
    memory_index = (
        _index_documents(term_memory, required_key="terms") if term_memory else {}
    )

    selected_ids = sorted(annotated_index)
    if document_ids is not None:
        selected_ids = [x for x in selected_ids if x in document_ids]
    if max_documents is not None:
        selected_ids = selected_ids[:max_documents]

    output = Path(output)
    experiment_dir = output / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_dir / "translation_evaluation_manifest.json"
    summary_path = experiment_dir / "translation_evaluation_summary.json"
    summary_csv_path = experiment_dir / "translation_evaluation_summary.csv"

    settings = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "experiment": experiment_name,
        "model": model,
        "base_url": base_url,
        "baseline_label": baseline_label,
        "controlled_label": controlled_label,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "position_control": position_control,
        "extra_body": extra_body or {},
        "term_memory": str(term_memory) if term_memory else None,
        "evaluation_scope": evaluation_scope,
        "non_term_sample_rate": non_term_sample_rate,
        "sample_seed": sample_seed,
        "json_mode": json_mode,
        "max_token_escalations": max_token_escalations,
    }
    evaluation_signature = _stable_hash(settings)

    manifest: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "in_progress",
        "experiment": experiment_name,
        "judge": {
            "model": model,
            "base_url": base_url,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "position_control": position_control,
            "json_mode": json_mode,
            "max_token_escalations": max_token_escalations,
        },
        "selection": {
            "evaluation_scope": evaluation_scope,
            "non_term_sample_rate": non_term_sample_rate,
            "sample_seed": sample_seed,
        },
        "inputs": {
            "annotated": str(annotated),
            "baseline": str(baseline),
            "controlled": str(controlled),
            "term_memory": str(term_memory) if term_memory else None,
        },
        "labels": {
            "baseline": baseline_label,
            "controlled": controlled_label,
        },
        "evaluation_signature": evaluation_signature,
        "documents_total": len(selected_ids),
        "documents": {},
        "missing_baseline_documents": [],
        "missing_controlled_documents": [],
        "updated_at": _now(),
    }
    if manifest_path.exists() and not overwrite:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("evaluation_signature") not in {None, evaluation_signature}:
            raise ValueError(
                "evaluation settings changed for this experiment; use a new "
                "--experiment name or --overwrite"
            )
        if isinstance(previous.get("documents"), dict):
            manifest["documents"] = previous["documents"]

    jobs: list[tuple[str, Path, Path, Path, Path | None, Path]] = []
    for document_id in selected_ids:
        baseline_path = baseline_index.get(document_id)
        controlled_path = controlled_index.get(document_id)
        if baseline_path is None:
            manifest["missing_baseline_documents"].append(document_id)
            continue
        if controlled_path is None:
            manifest["missing_controlled_documents"].append(document_id)
            continue
        jobs.append(
            (
                document_id,
                annotated_index[document_id],
                baseline_path,
                controlled_path,
                memory_index.get(document_id),
                experiment_dir / f"{_safe_slug(document_id)}.translation-evaluation.json",
            )
        )

    if not jobs:
        manifest["status"] = "error"
        manifest["error"] = (
            "no matched evaluation documents; check baseline/controlled paths "
            f"(missing baseline={len(manifest['missing_baseline_documents'])}, "
            f"missing controlled={len(manifest['missing_controlled_documents'])})"
        )
        manifest["updated_at"] = _now()
        atomic_write_json(manifest_path, manifest)
        raise ValueError(manifest["error"])

    key_env = api_key_env or (
        "DEEPSEEK_API_KEY" if "deepseek" in base_url.lower() else "OPENAI_API_KEY"
    )
    api_key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    timeout = httpx.Timeout(
        connect=min(30.0, timeout_seconds),
        read=timeout_seconds,
        write=min(120.0, timeout_seconds),
        pool=min(30.0, timeout_seconds),
    )
    limits = httpx.Limits(
        max_connections=max(16, workers * 2),
        max_keepalive_connections=max(8, workers),
    )
    http_client = httpx.Client(
        trust_env=False,
        timeout=timeout,
        limits=limits,
    )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )

    lock_path = experiment_dir / ".translation-evaluation.lock"
    progress = tqdm(
        total=len(jobs),
        desc=f"evaluate {experiment_name}",
        unit="doc",
        dynamic_ncols=True,
    )

    def process(job: tuple[str, Path, Path, Path, Path | None, Path]) -> tuple[str, dict[str, Any]]:
        document_id, annotated_path, baseline_path, controlled_path, memory_path, destination = job
        result = evaluate_translation_document(
            annotated_path,
            baseline_path,
            controlled_path,
            destination,
            client=client,
            experiment_name=experiment_name,
            model=model,
            base_url=base_url,
            evaluation_signature=evaluation_signature,
            term_memory_path=memory_path,
            baseline_label=baseline_label,
            controlled_label=controlled_label,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            position_control=position_control,
            save_raw=save_raw,
            overwrite=overwrite,
            max_paragraphs=max_paragraphs_per_document,
            extra_body=extra_body,
            evaluation_scope=evaluation_scope,
            non_term_sample_rate=non_term_sample_rate,
            sample_seed=sample_seed,
            json_mode=json_mode,
            max_token_escalations=max_token_escalations,
        )
        return document_id, result

    try:
        with exclusive_process_lock(lock_path, label=f"translation-evaluation:{experiment_name}"):
            if workers == 1:
                for job in jobs:
                    document_id = job[0]
                    try:
                        _, result = process(job)
                        manifest["documents"][document_id] = _manifest_document_row(job[-1], result)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # document-level failure must not stop the run
                        manifest["documents"][document_id] = {
                            "status": "error",
                            "output": str(job[-1]),
                            "error": f"{type(exc).__name__}: {exc}",
                            "updated_at": _now(),
                        }
                        tqdm.write(f"evaluation failed for {document_id}: {exc}")
                    finally:
                        progress.update(1)
                        manifest["updated_at"] = _now()
                        atomic_write_json(manifest_path, manifest)
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="legal-tc-evaluation",
                ) as executor:
                    futures: dict[
                        Future[tuple[str, dict[str, Any]]],
                        tuple[str, Path, Path, Path, Path | None, Path],
                    ] = {executor.submit(process, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        document_id = job[0]
                        try:
                            _, result = future.result()
                            manifest["documents"][document_id] = _manifest_document_row(
                                job[-1], result
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            manifest["documents"][document_id] = {
                                "status": "error",
                                "output": str(job[-1]),
                                "error": f"{type(exc).__name__}: {exc}",
                                "updated_at": _now(),
                            }
                            tqdm.write(f"evaluation failed for {document_id}: {exc}")
                        finally:
                            progress.update(1)
                            manifest["updated_at"] = _now()
                            atomic_write_json(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _now()
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        progress.close()
        http_client.close()

    summary = summarize_translation_evaluations(
        experiment_dir,
        baseline_label=baseline_label,
        controlled_label=controlled_label,
    )
    atomic_write_json(summary_path, summary)
    _write_summary_csv(summary_csv_path, summary)

    document_rows = list(manifest["documents"].values())
    document_errors = sum(1 for row in document_rows if row.get("status") == "error")
    paragraph_failures = int(summary.get("paragraphs_failed", 0))
    if (
        manifest["missing_baseline_documents"]
        or manifest["missing_controlled_documents"]
        or document_errors
    ):
        manifest["status"] = "partial"
    elif paragraph_failures:
        manifest["status"] = "complete_with_failures"
    else:
        manifest["status"] = "complete"
    manifest["summary"] = str(summary_path)
    manifest["summary_csv"] = str(summary_csv_path)
    manifest["totals"] = {
        "documents_requested": len(selected_ids),
        "documents_evaluated": int(summary.get("documents_evaluated", 0)),
        "document_errors": document_errors,
        "paragraphs_evaluated": int(summary.get("paragraphs_evaluated", 0)),
        "paragraphs_failed": paragraph_failures,
        "candidate_paragraphs": int(
            summary.get("selection", {}).get("candidate_paragraphs", 0)
        ),
        "selected_paragraphs": int(
            summary.get("selection", {}).get("selected_paragraphs", 0)
        ),
        "selected_term_bearing_paragraphs": int(
            summary.get("selection", {}).get(
                "selected_term_bearing_paragraphs", 0
            )
        ),
        "selected_non_term_paragraphs": int(
            summary.get("selection", {}).get("selected_non_term_paragraphs", 0)
        ),
        "prompt_tokens": int(summary.get("usage", {}).get("prompt_tokens", 0)),
        "completion_tokens": int(summary.get("usage", {}).get("completion_tokens", 0)),
        "total_tokens": int(summary.get("usage", {}).get("total_tokens", 0)),
    }
    manifest["updated_at"] = _now()
    atomic_write_json(manifest_path, manifest)

    return {
        "status": manifest["status"],
        "experiment_directory": str(experiment_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        **manifest["totals"],
    }


def evaluate_translation_document(
    annotated_path: Path,
    baseline_path: Path,
    controlled_path: Path,
    output_path: Path,
    *,
    client: Any,
    experiment_name: str,
    model: str,
    base_url: str,
    evaluation_signature: str,
    term_memory_path: Path | None = None,
    baseline_label: str = "baseline",
    controlled_label: str = "controlled",
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_retries: int = 2,
    position_control: str = "deterministic",
    save_raw: bool = False,
    overwrite: bool = False,
    max_paragraphs: int | None = None,
    extra_body: dict[str, Any] | None = None,
    evaluation_scope: str = "term-bearing-plus-sample",
    non_term_sample_rate: float = 0.1,
    sample_seed: int = 2026,
    json_mode: bool = True,
    max_token_escalations: int = 1,
) -> dict[str, Any]:
    annotated = _read_json_object(annotated_path)
    baseline = _read_json_object(baseline_path)
    controlled = _read_json_object(controlled_path)
    term_memory = _read_json_object(term_memory_path) if term_memory_path else None

    document_id = str(annotated.get("document_id") or annotated_path.stem)
    _validate_document_id(document_id, baseline, "baseline", baseline_path)
    _validate_document_id(document_id, controlled, "controlled", controlled_path)
    if term_memory is not None:
        _validate_document_id(document_id, term_memory, "term memory", term_memory_path)

    baseline_by_pid = _translations_by_pid(baseline)
    controlled_by_pid = _translations_by_pid(controlled)
    relevant_terms_by_pid = _terms_by_paragraph(annotated, term_memory)
    all_paragraphs = list(annotated.get("paragraphs") or [])
    paragraphs, selection = _select_paragraphs_for_evaluation(
        all_paragraphs,
        relevant_terms_by_pid,
        evaluation_scope=evaluation_scope,
        non_term_sample_rate=non_term_sample_rate,
        sample_seed=sample_seed,
        document_id=document_id,
    )
    selection["selected_before_pilot_cap"] = len(paragraphs)
    if max_paragraphs is not None:
        paragraphs = paragraphs[:max_paragraphs]
    selection["selected_paragraphs"] = len(paragraphs)
    selection["selected_term_bearing_paragraphs"] = sum(
        1
        for paragraph in paragraphs
        if relevant_terms_by_pid.get(str(paragraph.get("paragraph_id")))
    )
    selection["selected_non_term_paragraphs"] = (
        len(paragraphs) - selection["selected_term_bearing_paragraphs"]
    )

    existing: dict[str, Any] = {}
    if output_path.exists() and not overwrite:
        try:
            existing = _read_json_object(output_path)
        except Exception:
            existing = {}
        old_signature = existing.get("evaluation_signature")
        if old_signature not in {None, evaluation_signature}:
            raise ValueError(
                f"evaluation settings changed for {output_path}; use --overwrite "
                "or a new experiment name"
            )

    restored: dict[str, dict[str, Any]] = {}
    for row in existing.get("evaluations", []) if isinstance(existing, dict) else []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("paragraph_id"))
        if row.get("status") == "complete" and row.get("content_signature"):
            restored[pid] = row

    state: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "experiment": experiment_name,
        "document_id": document_id,
        "judge": {"model": model, "base_url": base_url},
        "labels": {
            "baseline": baseline_label,
            "controlled": controlled_label,
        },
        "evaluation_signature": evaluation_signature,
        "selection": selection,
        "inputs": {
            "annotated": str(annotated_path),
            "baseline": str(baseline_path),
            "controlled": str(controlled_path),
            "term_memory": str(term_memory_path) if term_memory_path else None,
        },
        "evaluations": [],
        "errors": [],
        "status": "in_progress",
        "started_at": existing.get("started_at") if existing else _now(),
        "updated_at": _now(),
    }

    ordered: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for paragraph in paragraphs:
        pid_value = paragraph.get("paragraph_id")
        pid = str(pid_value)
        source = str(paragraph.get("text", ""))
        baseline_translation = str(
            (baseline_by_pid.get(pid) or {}).get("translation", "")
        )
        controlled_translation = str(
            (controlled_by_pid.get(pid) or {}).get("translation", "")
        )
        relevant_terms = relevant_terms_by_pid.get(pid, [])
        content_signature = _stable_hash(
            {
                "source": source,
                "baseline": baseline_translation,
                "controlled": controlled_translation,
                "relevant_terms": relevant_terms,
            }
        )

        saved = restored.get(pid)
        if saved and saved.get("content_signature") == content_signature:
            ordered.append(saved)
            continue

        if not source or not baseline_translation or not controlled_translation:
            missing = []
            if not source:
                missing.append("source")
            if not baseline_translation:
                missing.append("baseline_translation")
            if not controlled_translation:
                missing.append("controlled_translation")
            error = {
                "paragraph_id": pid_value,
                "error_type": "MISSING_INPUT",
                "error": f"missing: {', '.join(missing)}",
                "updated_at": _now(),
            }
            errors.append(error)
            _write_document_progress(state, output_path, ordered, errors, paragraphs)
            continue

        try:
            judgment, passes, attempts, latency, usage = _judge_pair_robust(
                client=client,
                model=model,
                experiment_name=experiment_name,
                document_id=document_id,
                paragraph_id=pid,
                source=source,
                baseline_translation=baseline_translation,
                controlled_translation=controlled_translation,
                relevant_terms=relevant_terms,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                position_control=position_control,
                save_raw=save_raw,
                extra_body=extra_body,
                json_mode=json_mode,
                max_token_escalations=max_token_escalations,
            )
            row = {
                "paragraph_id": pid_value,
                "source": source,
                "baseline_translation": baseline_translation,
                "controlled_translation": controlled_translation,
                "baseline_label": baseline_label,
                "controlled_label": controlled_label,
                "has_benchmark_terms": bool(relevant_terms),
                "relevant_terms": relevant_terms,
                "content_signature": content_signature,
                "judgment": judgment,
                "passes": passes,
                "attempts": attempts,
                "latency_seconds": round(latency, 6),
                "usage": usage,
                "status": "complete",
                "updated_at": _now(),
            }
            ordered.append(row)
            errors = [x for x in errors if str(x.get("paragraph_id")) != pid]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error_row: dict[str, Any] = {
                "paragraph_id": pid_value,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "usage": getattr(exc, "usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }),
                "updated_at": _now(),
            }
            if save_raw and getattr(exc, "last_raw", ""):
                error_row["last_raw_output"] = getattr(exc, "last_raw")
            errors.append(error_row)
        _write_document_progress(state, output_path, ordered, errors, paragraphs)

    ordered.sort(key=lambda row: _paragraph_sort_key(row.get("paragraph_id")))
    state["evaluations"] = ordered
    state["errors"] = errors
    state["statistics"] = _document_statistics(paragraphs, ordered, errors)
    state["status"] = "complete" if not errors and len(ordered) == len(paragraphs) else "complete_with_failures"
    state["updated_at"] = _now()
    atomic_write_json(output_path, state)
    return state


def summarize_translation_evaluations(
    input_path: Path,
    *,
    baseline_label: str = "baseline",
    controlled_label: str = "controlled",
) -> dict[str, Any]:
    input_path = Path(input_path)
    files = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.rglob("*.translation-evaluation.json"))
    )
    rows: list[dict[str, Any]] = []
    failed = 0
    failure_usage_values: list[dict[str, Any]] = []
    documents = set()
    selection_totals = {
        "candidate_paragraphs": 0,
        "candidate_term_bearing_paragraphs": 0,
        "candidate_non_term_paragraphs": 0,
        "selected_paragraphs": 0,
        "selected_term_bearing_paragraphs": 0,
        "selected_non_term_paragraphs": 0,
    }
    for path in files:
        try:
            data = _read_json_object(path)
        except Exception:
            continue
        documents.add(str(data.get("document_id") or path.stem))
        rows.extend(x for x in data.get("evaluations", []) if isinstance(x, dict))
        selection = data.get("selection") or {}
        for key in selection_totals:
            selection_totals[key] += int(selection.get(key, 0) or 0)
        document_errors = [
            x for x in (data.get("errors", []) or []) if isinstance(x, dict)
        ]
        failed += len(document_errors)
        failure_usage_values.extend(x.get("usage") or {} for x in document_errors)

    successful_usage = _sum_usage(row.get("usage") for row in rows)
    total_usage = _merge_usage(successful_usage, _sum_usage(failure_usage_values))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "baseline_label": baseline_label,
        "controlled_label": controlled_label,
        "documents_evaluated": len(documents),
        "paragraphs_evaluated": len(rows),
        "paragraphs_failed": failed,
        "selection": selection_totals,
        "overall": _aggregate_rows(rows),
        "term_bearing": _aggregate_rows(
            [row for row in rows if row.get("has_benchmark_terms")]
        ),
        "non_term_bearing": _aggregate_rows(
            [row for row in rows if not row.get("has_benchmark_terms")]
        ),
        "usage": total_usage,
        "successful_usage": successful_usage,
        "failed_attempt_usage": _sum_usage(failure_usage_values),
        "updated_at": _now(),
    }


def _judge_pair_robust(
    *,
    client: Any,
    model: str,
    experiment_name: str,
    document_id: str,
    paragraph_id: str,
    source: str,
    baseline_translation: str,
    controlled_translation: str,
    relevant_terms: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    position_control: str,
    save_raw: bool,
    extra_body: dict[str, Any] | None,
    json_mode: bool,
    max_token_escalations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, float, dict[str, int]]:
    started = time.monotonic()
    orders: list[bool]
    if position_control == "fixed":
        orders = [False]
    elif position_control == "bidirectional":
        orders = [False, True]
    else:
        digest = hashlib.sha256(
            f"{experiment_name}\0{document_id}\0{paragraph_id}".encode("utf-8")
        ).digest()
        orders = [bool(digest[0] & 1)]

    passes: list[dict[str, Any]] = []
    total_attempts = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for swapped in orders:
        if swapped:
            translation_a = controlled_translation
            translation_b = baseline_translation
            role_a, role_b = "controlled", "baseline"
        else:
            translation_a = baseline_translation
            translation_b = controlled_translation
            role_a, role_b = "baseline", "controlled"
        prompt = _build_judge_messages(
            source=source,
            translation_a=translation_a,
            translation_b=translation_b,
            relevant_terms=relevant_terms,
        )
        parsed, raw, attempts, usage = _call_judge_with_retry(
            client=client,
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            extra_body=extra_body,
            json_mode=json_mode,
            max_token_escalations=max_token_escalations,
        )
        total_attempts += attempts
        total_usage = _merge_usage(total_usage, usage)
        mapped = _map_blind_judgment(parsed, role_a=role_a, role_b=role_b)
        pass_row: dict[str, Any] = {
            "order": {"A": role_a, "B": role_b},
            "judgment": mapped,
            "attempts": attempts,
            "usage": usage,
        }
        if save_raw:
            pass_row["raw_output"] = raw
        passes.append(pass_row)

    judgment = (
        passes[0]["judgment"]
        if len(passes) == 1
        else _combine_bidirectional_judgments(
            [x["judgment"] for x in passes]
        )
    )
    return (
        judgment,
        passes,
        total_attempts,
        time.monotonic() - started,
        total_usage,
    )


def _call_judge_with_retry(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    extra_body: dict[str, Any] | None,
    json_mode: bool,
    max_token_escalations: int,
) -> tuple[dict[str, Any], str, int, dict[str, int]]:
    last: Exception | None = None
    last_raw = ""
    cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    current_max_tokens = max_tokens
    token_escalations = 0
    use_json_mode = json_mode
    attempt = 0

    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": current_max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            cumulative_usage = _merge_usage(cumulative_usage, _response_usage(response))
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            last_raw = str(getattr(choice.message, "content", "") or "").strip()

            if finish_reason == "length":
                last = TranslationEvaluationError(
                    f"judge response was truncated at max_tokens={current_max_tokens}"
                )
                if token_escalations < max_token_escalations and attempt < max_retries:
                    token_escalations += 1
                    current_max_tokens *= 2
                    continue
                break

            parsed = _parse_json_object(last_raw)
            validated = _validate_blind_judgment(parsed)
            return validated, last_raw, attempt, cumulative_usage
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            if use_json_mode and _looks_like_unsupported_json_mode(exc):
                use_json_mode = False
                if attempt < max_retries:
                    continue
            if attempt < max_retries:
                time.sleep(
                    min(8.0, 0.5 * (2 ** (attempt - 1)))
                    + random.random() * 0.2
                )

    detail = f"; last output={last_raw[:500]!r}" if last_raw else ""
    raise TranslationEvaluationError(
        f"judge failed after {attempt} attempts: {type(last).__name__}: {last}{detail}",
        usage=cumulative_usage,
        last_raw=last_raw,
    )


def _looks_like_unsupported_json_mode(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "response_format" in text
        or "json_object" in text
        or ("json mode" in text and ("unsupported" in text or "not support" in text))
    )


def _build_judge_messages(
    *,
    source: str,
    translation_a: str,
    translation_b: str,
    relevant_terms: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a senior bilingual legal translator and a strict, impartial evaluator. "
        "Perform a blind pairwise evaluation of two Simplified-Chinese translations. "
        "Do not guess which system produced either translation. Assess each translation "
        "independently against the English source. Semantic and legal fidelity dominate; "
        "terminology compliance is secondary, and fluency is tertiary. Never reward a "
        "canonical term if it changes, narrows, broadens, or omits the legal meaning. "
        "Return exactly one JSON object and no prose outside it."
    )
    terms_text = "None supplied."
    if relevant_terms:
        lines = []
        for item in relevant_terms:
            source_term = str(item.get("source_term", "")).strip()
            canonical = str(item.get("canonical_translation", "")).strip()
            scope = str(item.get("legal_scope_summary", "")).strip()
            criticality = str(item.get("criticality", "")).strip()
            line = f"- {source_term} => {canonical}"
            if criticality:
                line += f" [{criticality}]"
            if scope:
                compact_scope = re.sub(r"\s+", " ", scope).strip()[:120]
                line += f"; scope: {compact_scope}"
            lines.append(line)
        terms_text = "\n".join(lines)

    user = f"""Evaluate the two translations below.

ENGLISH SOURCE:
{source}

APPROVED DOCUMENT-LEVEL TERMINOLOGY RELEVANT TO THIS PARAGRAPH:
{terms_text}

TRANSLATION A:
{translation_a}

TRANSLATION B:
{translation_b}

Score each translation from 1 to 5 on:
- semantic_fidelity: preservation of all propositions, conditions, exceptions, and relations.
- legal_accuracy: preservation of legal force, modality, parties, roles, rights, duties, scope, references, numbers, and dates.
- terminology_compliance: legally correct and document-consistent use of the approved terms; do not penalize an equivalent form when no approved term applies.
- fluency: professional, natural, readable legal Chinese without awkward or broken phrasing.
- overall_quality: holistic quality, giving priority to semantic_fidelity and legal_accuracy.

Critical-error flags are true only for a material error actually present in that translation.
Choose winner A, B, or TIE. Use TIE when the difference is not substantively meaningful.

Return this exact JSON shape:
{{
  "scores": {{
    "A": {{
      "semantic_fidelity": 1,
      "legal_accuracy": 1,
      "terminology_compliance": 1,
      "fluency": 1,
      "overall_quality": 1
    }},
    "B": {{
      "semantic_fidelity": 1,
      "legal_accuracy": 1,
      "terminology_compliance": 1,
      "fluency": 1,
      "overall_quality": 1
    }}
  }},
  "critical_errors": {{
    "A": {{
      "meaning_omission": false,
      "unsupported_addition": false,
      "polarity_or_modality_error": false,
      "party_or_role_error": false,
      "number_date_or_reference_error": false
    }},
    "B": {{
      "meaning_omission": false,
      "unsupported_addition": false,
      "polarity_or_modality_error": false,
      "party_or_role_error": false,
      "number_date_or_reference_error": false
    }}
  }},
  "winner": "A",
  "confidence": 0.0,
  "rationale": "Brief evidence-based explanation in Chinese, no more than 60 Chinese characters."
}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_blind_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("judge output missing scores")
    validated_scores: dict[str, dict[str, float]] = {}
    for side in ("A", "B"):
        raw = scores.get(side)
        if not isinstance(raw, dict):
            raise ValueError(f"judge output missing scores.{side}")
        validated_scores[side] = {}
        for field in SCORE_FIELDS:
            score = raw.get(field)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"scores.{side}.{field} must be numeric")
            score = float(score)
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"scores.{side}.{field} must be between 1 and 5")
            validated_scores[side][field] = score

    raw_errors = value.get("critical_errors")
    if not isinstance(raw_errors, dict):
        raise ValueError("judge output missing critical_errors")
    validated_errors: dict[str, dict[str, bool]] = {}
    for side in ("A", "B"):
        raw = raw_errors.get(side)
        if not isinstance(raw, dict):
            raise ValueError(f"judge output missing critical_errors.{side}")
        validated_errors[side] = {}
        for field in CRITICAL_ERROR_FIELDS:
            item = raw.get(field, False)
            if not isinstance(item, bool):
                raise ValueError(f"critical_errors.{side}.{field} must be boolean")
            validated_errors[side][field] = item

    winner = str(value.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError("winner must be A, B, or TIE")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rationale = str(value.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("rationale must not be empty")
    return {
        "scores": validated_scores,
        "critical_errors": validated_errors,
        "winner": winner,
        "confidence": confidence,
        "rationale": rationale,
    }


def _map_blind_judgment(
    judgment: dict[str, Any],
    *,
    role_a: str,
    role_b: str,
) -> dict[str, Any]:
    scores = {
        role_a: judgment["scores"]["A"],
        role_b: judgment["scores"]["B"],
    }
    critical_errors = {
        role_a: judgment["critical_errors"]["A"],
        role_b: judgment["critical_errors"]["B"],
    }
    winner_map = {"A": role_a, "B": role_b, "TIE": "tie"}
    return {
        "scores": scores,
        "critical_errors": critical_errors,
        "winner": winner_map[judgment["winner"]],
        "confidence": judgment["confidence"],
        "rationale": judgment["rationale"],
    }


def _combine_bidirectional_judgments(
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(judgments) != 2:
        raise ValueError("bidirectional aggregation requires exactly two judgments")
    combined_scores: dict[str, dict[str, float]] = {}
    combined_errors: dict[str, dict[str, bool]] = {}
    for role in ("baseline", "controlled"):
        combined_scores[role] = {
            field: round(
                sum(float(x["scores"][role][field]) for x in judgments) / 2.0,
                6,
            )
            for field in SCORE_FIELDS
        }
        combined_errors[role] = {
            field: all(bool(x["critical_errors"][role][field]) for x in judgments)
            for field in CRITICAL_ERROR_FIELDS
        }
    winners = [str(x.get("winner")) for x in judgments]
    winner = winners[0] if winners[0] == winners[1] else "tie"
    return {
        "scores": combined_scores,
        "critical_errors": combined_errors,
        "winner": winner,
        "confidence": round(
            sum(float(x.get("confidence", 0.0)) for x in judgments) / 2.0,
            6,
        ),
        "rationale": " | ".join(str(x.get("rationale", "")) for x in judgments),
        "position_disagreement": winners[0] != winners[1],
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "paragraphs": len(rows),
        "wins": {"baseline": 0, "controlled": 0, "tie": 0},
        "win_rates": {"baseline": 0.0, "controlled": 0.0, "tie": 0.0},
        "decisive_controlled_win_rate": 0.0,
        "scores": {},
        "score_deltas_controlled_minus_baseline": {},
        "critical_error_rates": {"baseline": {}, "controlled": {}},
        "position_disagreement_rate": 0.0,
    }
    if not rows:
        return result

    score_sums = {
        role: {field: 0.0 for field in SCORE_FIELDS}
        for role in ("baseline", "controlled")
    }
    error_counts = {
        role: {field: 0 for field in CRITICAL_ERROR_FIELDS}
        for role in ("baseline", "controlled")
    }
    position_disagreements = 0
    usable = 0
    for row in rows:
        judgment = row.get("judgment") or {}
        winner = str(judgment.get("winner", "tie"))
        if winner not in result["wins"]:
            winner = "tie"
        result["wins"][winner] += 1
        scores = judgment.get("scores") or {}
        errors = judgment.get("critical_errors") or {}
        try:
            for role in ("baseline", "controlled"):
                for field in SCORE_FIELDS:
                    score_sums[role][field] += float(scores[role][field])
                for field in CRITICAL_ERROR_FIELDS:
                    error_counts[role][field] += int(bool(errors[role][field]))
            usable += 1
        except (KeyError, TypeError, ValueError):
            continue
        if judgment.get("position_disagreement"):
            position_disagreements += 1

    total = len(rows)
    result["win_rates"] = {
        key: round(value / total, 6) for key, value in result["wins"].items()
    }
    decisive = result["wins"]["baseline"] + result["wins"]["controlled"]
    result["decisive_controlled_win_rate"] = (
        round(result["wins"]["controlled"] / decisive, 6) if decisive else 0.0
    )
    if usable:
        result["scores"] = {
            role: {
                field: round(score_sums[role][field] / usable, 6)
                for field in SCORE_FIELDS
            }
            for role in ("baseline", "controlled")
        }
        result["score_deltas_controlled_minus_baseline"] = {
            field: round(
                result["scores"]["controlled"][field]
                - result["scores"]["baseline"][field],
                6,
            )
            for field in SCORE_FIELDS
        }
        result["critical_error_rates"] = {
            role: {
                field: round(error_counts[role][field] / usable, 6)
                for field in CRITICAL_ERROR_FIELDS
            }
            for role in ("baseline", "controlled")
        }
    result["position_disagreement_rate"] = round(
        position_disagreements / total, 6
    )
    return result



def _select_paragraphs_for_evaluation(
    paragraphs: list[dict[str, Any]],
    relevant_terms_by_pid: dict[str, list[dict[str, Any]]],
    *,
    evaluation_scope: str,
    non_term_sample_rate: float,
    sample_seed: int,
    document_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    term_candidates = 0
    non_term_candidates = 0

    for paragraph in paragraphs:
        pid = str(paragraph.get("paragraph_id"))
        has_terms = bool(relevant_terms_by_pid.get(pid))
        if has_terms:
            term_candidates += 1
        else:
            non_term_candidates += 1

        include = evaluation_scope == "all" or has_terms
        if (
            not include
            and evaluation_scope == "term-bearing-plus-sample"
            and _deterministic_sample_hit(
                document_id=document_id,
                paragraph_id=pid,
                rate=non_term_sample_rate,
                seed=sample_seed,
            )
        ):
            include = True
        if include:
            selected.append(paragraph)

    return selected, {
        "evaluation_scope": evaluation_scope,
        "non_term_sample_rate": non_term_sample_rate,
        "sample_seed": sample_seed,
        "candidate_paragraphs": len(paragraphs),
        "candidate_term_bearing_paragraphs": term_candidates,
        "candidate_non_term_paragraphs": non_term_candidates,
    }


def _deterministic_sample_hit(
    *,
    document_id: str,
    paragraph_id: str,
    rate: float,
    seed: int,
) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(
        f"{seed}\0{document_id}\0{paragraph_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < rate

def _terms_by_paragraph(
    annotated: dict[str, Any],
    term_memory: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    memory_by_id = {
        str(term.get("term_id")): term
        for term in (term_memory or {}).get("terms", [])
        if isinstance(term, dict)
    }
    terms = annotated.get("benchmark_terms")
    if terms is None:
        terms = [
            term
            for term in annotated.get("terms", [])
            if (term.get("annotation") or {}).get("include_in_benchmark", False)
            and (term.get("annotation") or {}).get("criticality") != "EXCLUDED"
        ]
    by_pid: dict[str, dict[str, dict[str, Any]]] = {}
    for term in terms or []:
        term_id = str(term.get("term_id", ""))
        memory = memory_by_id.get(term_id, {})
        annotation = term.get("annotation") or {}
        item = {
            "term_id": term_id,
            "source_term": str(term.get("source_term", "")),
            "canonical_translation": str(
                memory.get("canonical_translation", "")
            ),
            "legal_scope_summary": str(memory.get("legal_scope_summary", "")),
            "criticality": str(
                memory.get("criticality") or annotation.get("criticality") or ""
            ),
        }
        for occurrence in term.get("occurrences", []) or []:
            pid = str(occurrence.get("paragraph_id"))
            by_pid.setdefault(pid, {})[term_id] = item
    return {
        pid: sorted(values.values(), key=lambda x: (x["term_id"], x["source_term"]))
        for pid, values in by_pid.items()
    }


def _translations_by_pid(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("paragraph_id")): item
        for item in data.get("translations", [])
        if isinstance(item, dict) and "paragraph_id" in item
    }


def _document_statistics(
    paragraphs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    usage = _merge_usage(
        _sum_usage(row.get("usage") for row in rows),
        _sum_usage(error.get("usage") for error in errors),
    )
    wins = {"baseline": 0, "controlled": 0, "tie": 0}
    for row in rows:
        winner = str((row.get("judgment") or {}).get("winner", "tie"))
        wins[winner if winner in wins else "tie"] += 1
    return {
        "paragraphs_total": len(paragraphs),
        "paragraphs_evaluated": len(rows),
        "paragraphs_failed": len(errors),
        "term_bearing_paragraphs": sum(
            1 for row in rows if row.get("has_benchmark_terms")
        ),
        "wins": wins,
        "usage": usage,
    }


def _write_document_progress(
    state: dict[str, Any],
    output_path: Path,
    ordered: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
) -> None:
    state["evaluations"] = sorted(
        ordered, key=lambda row: _paragraph_sort_key(row.get("paragraph_id"))
    )
    state["errors"] = errors
    state["statistics"] = _document_statistics(paragraphs, ordered, errors)
    state["updated_at"] = _now()
    atomic_write_json(output_path, state)


def _manifest_document_row(output_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "output": str(output_path),
        "statistics": result.get("statistics", {}),
        "selection": result.get("selection", {}),
        "updated_at": result.get("updated_at"),
    }


def _index_documents(path: Path | None, *, required_key: str) -> dict[str, Path]:
    if path is None:
        return {}
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.rglob("*.json"))
    result: dict[str, Path] = {}
    for candidate in candidates:
        if candidate.name.endswith(
            ("manifest.json", ".audit.json", ".checkpoint.json", "summary.json")
        ):
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or required_key not in value:
            continue
        document_id = str(value.get("document_id") or candidate.stem)
        if document_id in result:
            raise ValueError(
                f"duplicate document_id {document_id!r}: "
                f"{result[document_id]} and {candidate}"
            )
        result[document_id] = candidate
    return result


def _validate_document_id(
    expected: str,
    data: dict[str, Any],
    label: str,
    path: Path | None,
) -> None:
    actual = str(data.get("document_id") or "")
    if actual and actual != expected:
        raise ValueError(
            f"document_id mismatch for {label}: expected={expected!r}, "
            f"actual={actual!r}, path={path}"
        )


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(cleaned)):
            current = cleaned[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : end + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
    raise ValueError("could not parse JSON object from judge response")


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _sum_usage(values: Iterable[Any]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for value in values:
        if not isinstance(value, dict):
            continue
        total = _merge_usage(total, value)
    return total


def _merge_usage(left: dict[str, int], right: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(left.get(key, 0)) + int(right.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset",
        "paragraphs",
        "baseline_wins",
        "controlled_wins",
        "ties",
        "controlled_win_rate",
        "decisive_controlled_win_rate",
    ]
    for field in SCORE_FIELDS:
        fields.extend(
            [
                f"baseline_{field}",
                f"controlled_{field}",
                f"delta_{field}",
            ]
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subset in ("overall", "term_bearing", "non_term_bearing"):
            data = summary.get(subset) or {}
            wins = data.get("wins") or {}
            win_rates = data.get("win_rates") or {}
            scores = data.get("scores") or {}
            deltas = data.get("score_deltas_controlled_minus_baseline") or {}
            row: dict[str, Any] = {
                "subset": subset,
                "paragraphs": data.get("paragraphs", 0),
                "baseline_wins": wins.get("baseline", 0),
                "controlled_wins": wins.get("controlled", 0),
                "ties": wins.get("tie", 0),
                "controlled_win_rate": win_rates.get("controlled", 0.0),
                "decisive_controlled_win_rate": data.get(
                    "decisive_controlled_win_rate", 0.0
                ),
            }
            for field in SCORE_FIELDS:
                row[f"baseline_{field}"] = (scores.get("baseline") or {}).get(field)
                row[f"controlled_{field}"] = (scores.get("controlled") or {}).get(field)
                row[f"delta_{field}"] = deltas.get(field)
            writer.writerow(row)


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "document"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paragraph_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return 10**18, str(value)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
