from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm

from .checkpoint import Checkpoint

FINAL_STATUSES = {
    "FOUND",
    "OMITTED",
    "IMPLICIT",
    "AMBIGUOUS",
    "TRANSLATION_ERROR",
}

SYSTEM_PROMPT = """You are a bilingual legal-translation alignment annotator.
For each occurrence, identify how the specified English source term is rendered
in the supplied Simplified Chinese target paragraph.

This is extraction, not rewriting.

Rules:
1. If an explicit Chinese expression corresponds to the source term, return the
   exact verbatim substring from target_paragraph and status FOUND.
2. Do not invent, normalize, paraphrase, or improve the target text.
3. If the meaning is present only implicitly and no exact span can be extracted,
   use IMPLICIT and rendering=null.
4. If the source term is omitted, use OMITTED and rendering=null.
5. If several target spans are plausible and no unique span can be selected,
   use AMBIGUOUS and rendering=null.
6. If the target contains an evident mistranslation but a concrete target span
   can still be identified, use TRANSLATION_ERROR and return that exact span.
7. Return one result for every occurrence_id, in the same order.
8. Return JSON only, with no Markdown.
"""


def extract_renderings_with_llm(
    input_path: Path,
    output_path: Path,
    *,
    model: str = "gpt-oss-120b",
    base_url: str = "http://10.12.143.51:8000/v1",
    batch_size: int = 1,
    workers: int = 1,
    confidence_threshold: float = 0.8,
    max_retries: int = 5,
    timeout_seconds: float = 300.0,
    checkpoint_path: Path | None = None,
    overwrite: bool = False,
    max_output_tokens: int = 384,
    max_source_chars: int = 700,
    max_target_chars: int = 2200,
    estimate_only: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be >= 1")
    if max_source_chars < 1:
        raise ValueError("max_source_chars must be >= 1")
    if max_target_chars < 1:
        raise ValueError("max_target_chars must be >= 1")

    documents = _load_review_documents(input_path)
    is_multi = input_path.is_dir()
    if is_multi:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    _merge_existing_outputs(documents, output_path, is_multi=is_multi)
    all_rows = [row for doc in documents for row in doc["rows"]]
    pending = [
        row
        for row in all_rows
        if overwrite or row.get("extraction_status") not in FINAL_STATUSES
    ]
    occurrence_to_document = {
        row["occurrence_id"]: document
        for document in documents
        for row in document["rows"]
    }

    if checkpoint_path is None:
        checkpoint_path = (
            output_path / "renderings.checkpoint.json"
            if is_multi
            else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
        )
    audit_path = (
        output_path / "renderings.audit.json"
        if is_multi
        else output_path.with_suffix(output_path.suffix + ".audit.json")
    )

    checkpoint = Checkpoint(checkpoint_path)
    batch_specs: list[dict[str, Any]] = []
    restored_usage = _empty_observed_usage()
    restored_batches = 0

    for sequence, start in enumerate(range(0, len(pending), batch_size)):
        batch = pending[start : start + batch_size]
        signature = _batch_signature(batch)
        checkpoint_key = _checkpoint_key(signature)
        payload = _build_payload(
            batch,
            max_source_chars=max_source_chars,
            max_target_chars=max_target_chars,
        )
        estimated_prompt_tokens = _estimate_request_tokens(payload)
        spec = {
            "sequence": sequence,
            "checkpoint_key": checkpoint_key,
            "signature": signature,
            "batch": batch,
            "payload": payload,
            "estimated_prompt_tokens": estimated_prompt_tokens,
        }

        saved = None if overwrite else checkpoint.get(checkpoint_key)
        if saved:
            meta = saved.get("meta", {})
            if meta.get("signature") == signature:
                _apply_saved_results(all_rows, meta.get("results", []))
                _merge_observed_usage(
                    restored_usage,
                    meta.get("token_usage", {}),
                )
                restored_batches += 1
                continue
            checkpoint.remove(checkpoint_key)
        batch_specs.append(spec)

    estimated_prompt_tokens = sum(
        int(spec["estimated_prompt_tokens"])
        for spec in batch_specs
    )
    token_estimate = {
        "estimator": "approx_cjk_1_token_each_plus_other_chars_div_4",
        "pending_occurrences": len(pending),
        "pending_batches_to_request": len(batch_specs),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "maximum_completion_token_budget": len(batch_specs) * max_output_tokens,
        "estimated_max_total_tokens": (
            estimated_prompt_tokens + len(batch_specs) * max_output_tokens
        ),
    }
    run_usage = _empty_observed_usage()

    tqdm.write(
        "Token estimate for remaining requests: "
        f"prompt≈{token_estimate['estimated_prompt_tokens']:,}, "
        f"completion_budget≤{token_estimate['maximum_completion_token_budget']:,}, "
        f"max_total≈{token_estimate['estimated_max_total_tokens']:,}"
    )

    if estimate_only:
        return _write_audit(
            audit_path,
            input_path=input_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            model=model,
            base_url=base_url,
            batch_size=batch_size,
            workers=workers,
            confidence_threshold=confidence_threshold,
            max_output_tokens=max_output_tokens,
            max_source_chars=max_source_chars,
            max_target_chars=max_target_chars,
            rows=all_rows,
            status="estimated",
            token_estimate=token_estimate,
            run_usage=run_usage,
            restored_usage=restored_usage,
            restored_batches=restored_batches,
        )

    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(
            connect=min(30.0, timeout_seconds),
            read=timeout_seconds,
            write=min(120.0, timeout_seconds),
            pool=min(30.0, timeout_seconds),
        ),
        limits=httpx.Limits(
            max_connections=max(4, workers * 2),
            max_keepalive_connections=max(2, workers),
        ),
    )
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )

    completed_now = 0
    failures: list[dict[str, Any]] = []
    low_confidence_count = sum(
        1
        for row in all_rows
        if _is_low_confidence(row, confidence_threshold)
    )
    total_batches = len(batch_specs) + restored_batches
    progress = tqdm(total=total_batches, desc="extract renderings")
    if restored_batches:
        progress.update(restored_batches)

    executor: ThreadPoolExecutor | None = None
    fatal_error: BaseException | None = None
    try:
        if batch_specs:
            executor = ThreadPoolExecutor(
                max_workers=min(workers, len(batch_specs)),
                thread_name_prefix="rendering",
            )
            future_to_spec: dict[Future[Any], dict[str, Any]] = {
                executor.submit(
                    _call_and_validate,
                    client,
                    model=model,
                    payload=spec["payload"],
                    batch=spec["batch"],
                    max_retries=max_retries,
                    max_output_tokens=max_output_tokens,
                ): spec
                for spec in batch_specs
            }

            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                batch = spec["batch"]
                try:
                    parsed, raw, token_usage = future.result()
                except Exception as exc:
                    token_usage = getattr(exc, "token_usage", None)
                    if isinstance(token_usage, dict):
                        _merge_observed_usage(run_usage, token_usage)
                    failures.append({
                        "checkpoint_key": spec["checkpoint_key"],
                        "occurrence_ids": [row["occurrence_id"] for row in batch],
                        "error": str(exc),
                    })
                    progress.update(1)
                    progress.set_postfix(
                        completed=sum(
                            1 for row in all_rows
                            if row.get("extraction_status") in FINAL_STATUSES
                        ),
                        failed=len(failures),
                        tokens=run_usage["total_tokens"],
                    )
                    continue

                _merge_observed_usage(run_usage, token_usage)
                saved_results: list[dict[str, Any]] = []
                by_id = {row["occurrence_id"]: row for row in batch}
                changed_documents: dict[Path, dict[str, Any]] = {}
                for result in parsed["results"]:
                    row = by_id[result["occurrence_id"]]
                    row["rendering"] = result["rendering"]
                    row["extraction_status"] = result["status"]
                    row["extraction_confidence"] = result["confidence"]
                    row["extraction_rationale"] = result["rationale"]
                    row["extraction_model"] = model
                    if result["confidence"] < confidence_threshold:
                        row["notes"] = _append_note(
                            row.get("notes"),
                            f"LOW_CONFIDENCE<{confidence_threshold}",
                        )
                        low_confidence_count += 1
                    saved_results.append({
                        key: row.get(key)
                        for key in (
                            "occurrence_id",
                            "rendering",
                            "extraction_status",
                            "extraction_confidence",
                            "extraction_rationale",
                            "extraction_model",
                            "notes",
                        )
                    })
                    document = occurrence_to_document[row["occurrence_id"]]
                    changed_documents[document["path"]] = document

                checkpoint.mark(spec["checkpoint_key"], {
                    "signature": spec["signature"],
                    "results": saved_results,
                    "raw_output": raw,
                    "estimated_prompt_tokens": spec["estimated_prompt_tokens"],
                    "token_usage": token_usage,
                })
                completed_now += 1
                _write_review_outputs(
                    list(changed_documents.values()),
                    output_path,
                    is_multi=is_multi,
                )
                _write_audit(
                    audit_path,
                    input_path=input_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                    model=model,
                    base_url=base_url,
                    batch_size=batch_size,
                    workers=workers,
                    confidence_threshold=confidence_threshold,
                    max_output_tokens=max_output_tokens,
                    max_source_chars=max_source_chars,
                    max_target_chars=max_target_chars,
                    rows=all_rows,
                    status="in_progress",
                    completed_now=completed_now,
                    token_estimate=token_estimate,
                    run_usage=run_usage,
                    restored_usage=restored_usage,
                    restored_batches=restored_batches,
                    failures=failures,
                )
                progress.update(1)
                progress.set_postfix(
                    completed=sum(
                        1 for row in all_rows
                        if row.get("extraction_status") in FINAL_STATUSES
                    ),
                    failed=len(failures),
                    tokens=run_usage["total_tokens"],
                )
    except BaseException as exc:
        fatal_error = exc
    finally:
        if executor is not None:
            executor.shutdown(
                wait=fatal_error is None,
                cancel_futures=fatal_error is not None,
            )
        progress.close()
        client.close()
        http_client.close()

    _write_review_outputs(documents, output_path, is_multi=is_multi)

    if fatal_error is not None:
        _write_audit(
            audit_path,
            input_path=input_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            model=model,
            base_url=base_url,
            batch_size=batch_size,
            workers=workers,
            confidence_threshold=confidence_threshold,
            max_output_tokens=max_output_tokens,
            max_source_chars=max_source_chars,
            max_target_chars=max_target_chars,
            rows=all_rows,
            status="failed",
            error=str(fatal_error),
            completed_now=completed_now,
            token_estimate=token_estimate,
            run_usage=run_usage,
            restored_usage=restored_usage,
            restored_batches=restored_batches,
            failures=failures,
        )
        raise fatal_error

    if failures:
        error = (
            f"{len(failures)} rendering batch(es) failed. "
            "Successful batches were checkpointed; rerun the same command to retry failures."
        )
        result = _write_audit(
            audit_path,
            input_path=input_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            model=model,
            base_url=base_url,
            batch_size=batch_size,
            workers=workers,
            confidence_threshold=confidence_threshold,
            max_output_tokens=max_output_tokens,
            max_source_chars=max_source_chars,
            max_target_chars=max_target_chars,
            rows=all_rows,
            status="partial_failed",
            error=error,
            completed_now=completed_now,
            token_estimate=token_estimate,
            run_usage=run_usage,
            restored_usage=restored_usage,
            restored_batches=restored_batches,
            failures=failures,
        )
        raise RuntimeError(f"{error} Audit: {audit_path}")

    return _write_audit(
        audit_path,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        model=model,
        base_url=base_url,
        batch_size=batch_size,
        workers=workers,
        confidence_threshold=confidence_threshold,
        max_output_tokens=max_output_tokens,
        max_source_chars=max_source_chars,
        max_target_chars=max_target_chars,
        rows=all_rows,
        status="complete",
        completed_now=completed_now,
        token_estimate=token_estimate,
        run_usage=run_usage,
        restored_usage=restored_usage,
        restored_batches=restored_batches,
    )


def _load_review_documents(input_path: Path) -> list[dict[str, Any]]:
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.json"))
    documents: list[dict[str, Any]] = []
    seen_occurrence_ids: set[str] = set()
    for path in paths:
        if path.name.endswith(("manifest.json", ".audit.json", ".checkpoint.json")):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Review row {index} in {path} is not an object")
            occurrence_id = str(row.get("occurrence_id") or "")
            if not occurrence_id:
                occurrence_id = (
                    f"{row.get('document_id', path.stem)}::{row.get('term_id')}::"
                    f"{row.get('occurrence_index', index)}::{row.get('paragraph_id')}"
                )
                row["occurrence_id"] = occurrence_id
            if occurrence_id in seen_occurrence_ids:
                raise ValueError(f"Duplicate occurrence_id: {occurrence_id}")
            seen_occurrence_ids.add(occurrence_id)
        documents.append({"path": path, "rows": rows})
    if not documents:
        raise ValueError(f"No occurrence review JSON files found in {input_path}")
    return documents


def _merge_existing_outputs(
    documents: list[dict[str, Any]],
    output_path: Path,
    *,
    is_multi: bool,
) -> None:
    fields = (
        "rendering",
        "extraction_status",
        "extraction_confidence",
        "extraction_rationale",
        "extraction_model",
        "notes",
    )
    for document in documents:
        existing_path = (
            output_path / document["path"].name
            if is_multi
            else output_path
        )
        if not existing_path.exists():
            continue
        try:
            existing_rows = json.loads(existing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        existing_by_id = {
            str(row.get("occurrence_id")): row
            for row in existing_rows
            if isinstance(row, dict) and row.get("occurrence_id")
        }
        for row in document["rows"]:
            existing = existing_by_id.get(row["occurrence_id"])
            if not existing:
                continue
            for field in fields:
                if field in existing:
                    row[field] = existing[field]


def _write_review_outputs(
    documents: list[dict[str, Any]],
    output_path: Path,
    *,
    is_multi: bool,
) -> None:
    for document in documents:
        destination = (
            output_path / document["path"].name
            if is_multi
            else output_path
        )
        _atomic_json(destination, document["rows"])


def _truncate_text(value: Any, max_chars: int) -> str:
    """Limit prompt size while retaining both ends of a long paragraph."""
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    if max_chars <= 40:
        return text[:max_chars]
    marker = "\n...[TRUNCATED]...\n"
    remaining = max_chars - len(marker)
    left = remaining // 2
    right = remaining - left
    return text[:left] + marker + text[-right:]


def _format_exception(exc: Exception) -> str:
    """Include the HTTP response body when the OpenAI client exposes it."""
    parts = [f"{type(exc).__name__}: {exc}"]
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            parts.append(f"status_code={status_code}")
        try:
            body = response.text
        except Exception:
            body = None
        if body:
            parts.append(f"response_body={body[:2000]}")
    return " | ".join(parts)


def _build_payload(
    batch: list[dict[str, Any]],
    *,
    max_source_chars: int,
    max_target_chars: int,
) -> dict[str, Any]:
    return {
        "task": "extract_legal_term_renderings",
        "items": [
            {
                "occurrence_id": row["occurrence_id"],
                "source_term": row["source_term"],
                "source_context": _truncate_text(
                    row.get("source_context", ""),
                    max_source_chars,
                ),
                "target_paragraph": _truncate_text(
                    row.get("target_paragraph", ""),
                    max_target_chars,
                ),
            }
            for row in batch
        ],
        "output_schema": {
            "results": [
                {
                    "occurrence_id": "string",
                    "rendering": "exact target substring or null",
                    "status": "FOUND|OMITTED|IMPLICIT|AMBIGUOUS|TRANSLATION_ERROR",
                    "confidence": "number from 0 to 1",
                    "rationale": "brief explanation",
                }
            ]
        },
    }


def _checkpoint_key(signature: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "batch-" + hashlib.sha256(encoded).hexdigest()[:24]


def _estimate_request_tokens(payload: dict[str, Any]) -> int:
    user_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # A small allowance covers chat-role wrappers and message separators.
    return _estimate_text_tokens(SYSTEM_PROMPT) + _estimate_text_tokens(user_text) + 24


def _estimate_text_tokens(text: str) -> int:
    """Conservative local estimate without downloading a model tokenizer."""
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x2FA1F
        ):
            cjk += 1
        else:
            other += 1
    return cjk + math.ceil(other / 4)


def _empty_observed_usage() -> dict[str, int]:
    return {
        "attempt_count": 0,
        "api_reported_attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _merge_observed_usage(total: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        try:
            total[key] += int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue


def _extract_response_usage(response: Any) -> dict[str, int]:
    result = _empty_observed_usage()
    result["attempt_count"] = 1
    usage = getattr(response, "usage", None)
    if usage is None:
        return result
    result["api_reported_attempts"] = 1
    for source, destination in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, source, 0)
        result[destination] = int(value or 0)
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        result["reasoning_tokens"] = int(
            getattr(details, "reasoning_tokens", 0) or 0
        )
    return result


def _call_and_validate(
    client: Any,
    *,
    model: str,
    payload: dict[str, Any],
    batch: list[dict[str, Any]],
    max_retries: int,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    last_error: Exception | None = None
    cumulative_usage = _empty_observed_usage()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_output_tokens,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            )
            _merge_observed_usage(
                cumulative_usage,
                _extract_response_usage(response),
            )
            raw = response.choices[0].message.content or ""
            parsed = _parse_json(raw)
            validated = _validate_results(parsed, batch)
            return validated, raw, cumulative_usage
        except Exception as exc:  # retry network, JSON and validation failures
            # Network failures have no response usage. Validation failures after a
            # successful response were already counted above.
            if cumulative_usage["attempt_count"] <= attempt:
                cumulative_usage["attempt_count"] += 1
            last_error = RuntimeError(_format_exception(exc))
            if attempt + 1 >= max_retries:
                break
            time.sleep(min(60.0, 2.0 ** attempt))
    error = RuntimeError(
        f"LLM rendering extraction failed after {max_retries} attempts: {last_error}"
    )
    error.token_usage = cumulative_usage  # type: ignore[attr-defined]
    raise error from last_error


def _parse_json(raw: str) -> Any:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_obj, end_obj = text.find("{"), text.rfind("}")
        if 0 <= start_obj < end_obj:
            return json.loads(text[start_obj : end_obj + 1])
        start_arr, end_arr = text.find("["), text.rfind("]")
        if 0 <= start_arr < end_arr:
            return json.loads(text[start_arr : end_arr + 1])
        raise


def _validate_results(parsed: Any, batch: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(parsed, list):
        parsed = {"results": parsed}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise ValueError("LLM output must be an object containing a results array")
    results = parsed["results"]
    expected_ids = [row["occurrence_id"] for row in batch]
    returned_ids = [str(item.get("occurrence_id", "")) for item in results]
    if returned_ids != expected_ids:
        raise ValueError(
            "LLM occurrence_id sequence does not match input: "
            f"expected={expected_ids}, returned={returned_ids}"
        )

    by_id = {row["occurrence_id"]: row for row in batch}
    validated: list[dict[str, Any]] = []
    for item in results:
        occurrence_id = str(item["occurrence_id"])
        status = str(item.get("status", "")).upper()
        if status not in FINAL_STATUSES:
            raise ValueError(f"Invalid extraction status for {occurrence_id}: {status}")
        confidence = float(item.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid confidence for {occurrence_id}: {confidence}")
        rendering = item.get("rendering")
        if rendering is not None:
            rendering = str(rendering).strip()
        target = str(by_id[occurrence_id].get("target_paragraph", ""))
        if status in {"FOUND", "TRANSLATION_ERROR"}:
            if not rendering:
                raise ValueError(f"{status} requires rendering for {occurrence_id}")
            if rendering not in target:
                raise ValueError(
                    f"Rendering is not an exact target substring for {occurrence_id}: "
                    f"{rendering!r}"
                )
        elif rendering:
            raise ValueError(f"{status} requires rendering=null for {occurrence_id}")
        validated.append({
            "occurrence_id": occurrence_id,
            "rendering": rendering,
            "status": status,
            "confidence": confidence,
            "rationale": str(item.get("rationale", "")).strip(),
        })
    return {"results": validated}


def _batch_signature(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "occurrence_id": row["occurrence_id"],
            "source_term": row.get("source_term"),
            "target_paragraph": row.get("target_paragraph"),
        }
        for row in batch
    ]


def _apply_saved_results(
    rows: list[dict[str, Any]],
    saved_results: list[dict[str, Any]],
) -> None:
    by_id = {row["occurrence_id"]: row for row in rows}
    for saved in saved_results:
        row = by_id.get(str(saved.get("occurrence_id")))
        if not row:
            continue
        for key, value in saved.items():
            if key != "occurrence_id":
                row[key] = value


def _write_audit(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model: str,
    base_url: str,
    batch_size: int,
    workers: int,
    confidence_threshold: float,
    max_output_tokens: int,
    max_source_chars: int,
    max_target_chars: int,
    rows: list[dict[str, Any]],
    status: str,
    token_estimate: dict[str, Any],
    run_usage: dict[str, int],
    restored_usage: dict[str, int],
    restored_batches: int,
    error: str | None = None,
    completed_now: int = 0,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status_counts = Counter(
        row.get("extraction_status")
        for row in rows
        if row.get("extraction_status")
    )
    completed = sum(status_counts.values())
    low_confidence = sum(
        1 for row in rows if _is_low_confidence(row, confidence_threshold)
    )
    cumulative_usage = _empty_observed_usage()
    _merge_observed_usage(cumulative_usage, restored_usage)
    _merge_observed_usage(cumulative_usage, run_usage)
    audit: dict[str, Any] = {
        "status": status,
        "input": str(input_path),
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "model": model,
        "base_url": base_url,
        "batch_size": batch_size,
        "workers": workers,
        "confidence_threshold": confidence_threshold,
        "max_output_tokens": max_output_tokens,
        "max_source_chars": max_source_chars,
        "max_target_chars": max_target_chars,
        "occurrence_count": len(rows),
        "completed_count": completed,
        "remaining_count": len(rows) - completed,
        "low_confidence_count": low_confidence,
        "status_counts": dict(status_counts),
        "completed_batches_this_run": completed_now,
        "restored_batches_from_checkpoint": restored_batches,
        "token_usage": {
            "estimate_for_requests_pending_at_start": token_estimate,
            "this_run_api_usage": run_usage,
            "restored_checkpoint_api_usage": restored_usage,
            "cumulative_accounted_api_usage": cumulative_usage,
            "note": (
                "API usage is exact when vLLM returns response.usage. "
                "The pre-run estimate is approximate and the completion figure is a maximum budget."
            ),
        },
    }
    if failures:
        audit["failed_batch_count"] = len(failures)
        audit["failed_batches"] = failures[:100]
    if error:
        audit["error"] = error
    _atomic_json(path, audit)
    return audit


def _is_low_confidence(row: dict[str, Any], threshold: float) -> bool:
    confidence = row.get("extraction_confidence")
    return confidence is not None and float(confidence) < threshold


def _append_note(existing: Any, value: str) -> str:
    text = str(existing or "").strip()
    if not text:
        return value
    if value in text:
        return text
    return f"{text}; {value}"


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)
