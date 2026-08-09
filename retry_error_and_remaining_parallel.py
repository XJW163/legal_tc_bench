#!/usr/bin/env python3
"""Parallel, context-free completion of unfinished LegalTC-Bench paragraphs.

The script reads an existing ``*.translation.json`` checkpoint and its source
structured dataset. It translates every paragraph that does not already have a
non-empty successful translation. This includes:

* paragraphs currently listed in ``output["errors"]``;
* paragraphs never attempted and absent from both translations and errors.

Only unfinished paragraphs are submitted to the model. Existing successful
translations are preserved. Results are written atomically back to the same
output file after every completed task, so rerunning the script safely resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from legal_tc_bench.translate import (
    LATIN_LETTER_RE,
    PROMPT_PROFILES,
    ZERO_ENGLISH_PROFILES,
    _atomic_json,
    _clean_translation,
    _now,
    _text_hash,
    _translate_source_text,
    contains_latin_letters,
    find_latin_residues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel-translate all unfinished paragraphs in an existing "
            "LegalTC-Bench output JSON, without document context."
        )
    )
    parser.add_argument("output", type=Path, help="Existing *.translation.json checkpoint")
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Source structured dataset JSON. If omitted, output.source_path is used."
        ),
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel paragraph requests")
    parser.add_argument("--model", help="Override output.experiment.model")
    parser.add_argument("--base-url", help="Override output.experiment.base_url")
    parser.add_argument(
        "--prompt-profile",
        choices=sorted(PROMPT_PROFILES),
        help="Override output.experiment.prompt_profile",
    )
    parser.add_argument("--temperature", type=float, help="Override recorded temperature")
    parser.add_argument("--max-tokens", type=int, help="Override recorded max_tokens")
    parser.add_argument(
        "--source-chunk-chars",
        type=int,
        help="Override recorded source_chunk_chars",
    )
    parser.add_argument(
        "--extra-body-json",
        help="Optional JSON object passed to chat.completions",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Attempts used by the package translator; must be at least 1",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create <output>.bak before modification",
    )
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return data


def choose(value: Any, fallback: Any, default: Any) -> Any:
    if value is not None:
        return value
    if fallback is not None:
        return fallback
    return default


def paragraph_sort_key(item: dict[str, Any]) -> tuple[int, Any]:
    value = item.get("paragraph_id")
    try:
        return 0, int(str(value))
    except (TypeError, ValueError):
        return 1, str(value)


def successful_translation(item: Any) -> bool:
    return isinstance(item, dict) and bool(str(item.get("translation", "")).strip())


def update_statistics(
    state: dict[str, Any],
    *,
    total_paragraphs: int,
    newly_translated: int,
) -> None:
    translations = [x for x in state.get("translations", []) if isinstance(x, dict)]
    errors = [x for x in state.get("errors", []) if isinstance(x, dict)]
    old = state.get("statistics") if isinstance(state.get("statistics"), dict) else {}

    translated_ids = {
        str(x.get("paragraph_id"))
        for x in translations
        if x.get("paragraph_id") is not None and successful_translation(x)
    }
    failed_ids = {
        str(x.get("paragraph_id"))
        for x in errors
        if x.get("paragraph_id") is not None and str(x.get("paragraph_id")) not in translated_ids
    }
    nonempty = sum(1 for x in translations if successful_translation(x))
    empty = len(translations) - nonempty

    state["statistics"] = {
        **old,
        "total_paragraphs": total_paragraphs,
        "translated_paragraphs": len(translated_ids),
        "nonempty_translations": nonempty,
        "empty_translations": empty,
        "failed_paragraphs": len(failed_ids),
        # Matches the original package convention: every not-yet-successful paragraph.
        "remaining_paragraphs": max(0, total_paragraphs - len(translated_ids)),
        "newly_translated_paragraphs": newly_translated,
    }
    state["status"] = (
        "complete"
        if len(translated_ids) == total_paragraphs and not failed_ids
        else "incomplete"
    )
    state["updated_at"] = _now()


class ClientFactory:
    """Create one OpenAI/httpx client pair per worker thread."""

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.local = threading.local()
        self.lock = threading.Lock()
        self.http_clients: list[httpx.Client] = []

    def get(self) -> Any:
        client = getattr(self.local, "openai_client", None)
        if client is not None:
            return client

        from openai import OpenAI

        timeout = httpx.Timeout(
            connect=min(30.0, self.timeout_seconds),
            read=self.timeout_seconds,
            write=min(120.0, self.timeout_seconds),
            pool=min(30.0, self.timeout_seconds),
        )
        http_client = httpx.Client(trust_env=False, timeout=timeout)
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=self.base_url,
            http_client=http_client,
            max_retries=0,
        )
        self.local.openai_client = client
        with self.lock:
            self.http_clients.append(http_client)
        return client

    def close(self) -> None:
        for http_client in self.http_clients:
            http_client.close()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    output_path = args.output.expanduser().resolve()
    state = load_json_object(output_path, "output")
    if not isinstance(state.get("translations", []), list):
        raise ValueError(f"{output_path}: translations must be a list")
    if not isinstance(state.get("errors", []), list):
        raise ValueError(f"{output_path}: errors must be a list")

    if args.dataset is not None:
        dataset_path = args.dataset.expanduser().resolve()
    else:
        recorded_source = state.get("source_path")
        if not recorded_source:
            raise ValueError("output has no source_path; provide --dataset")
        candidate = Path(str(recorded_source)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        dataset_path = candidate

    dataset = load_json_object(dataset_path, "dataset")
    paragraphs = dataset.get("paragraphs")
    if not isinstance(paragraphs, list):
        raise ValueError(f"{dataset_path}: missing paragraphs list")

    output_document_id = str(state.get("document_id", ""))
    dataset_document_id = str(dataset.get("document_id") or dataset_path.stem)
    if output_document_id and output_document_id != dataset_document_id:
        raise ValueError(
            f"document_id mismatch: output={output_document_id!r}, "
            f"dataset={dataset_document_id!r}"
        )

    experiment = state.get("experiment") if isinstance(state.get("experiment"), dict) else {}
    model = str(choose(args.model, experiment.get("model"), "gpt-oss-120b"))
    base_url = str(choose(args.base_url, experiment.get("base_url"), "http://10.12.143.51:8000/v1"))
    prompt_profile = str(choose(args.prompt_profile, experiment.get("prompt_profile"), "baseline-v1"))
    temperature = float(choose(args.temperature, experiment.get("temperature"), 0.0))
    max_tokens = int(choose(args.max_tokens, experiment.get("max_tokens"), 4096))
    source_chunk_chars = int(choose(args.source_chunk_chars, experiment.get("source_chunk_chars"), 12000))

    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile: {prompt_profile}")

    recorded_extra_body = experiment.get("extra_body")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else recorded_extra_body
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a JSON object")

    # Keep exactly one successful record per paragraph ID.
    translations_by_id: dict[str, dict[str, Any]] = {}
    for item in state.get("translations", []):
        if not isinstance(item, dict) or item.get("paragraph_id") is None:
            continue
        pid = str(item.get("paragraph_id"))
        if successful_translation(item):
            translations_by_id[pid] = item

    # Every paragraph without a successful translation is work to do. This set
    # naturally contains both existing errors and never-attempted paragraphs.
    tasks: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            continue
        pid_value = paragraph.get("paragraph_id")
        pid = str(pid_value)
        if pid in translations_by_id:
            continue
        tasks.append({
            "paragraph_id": pid_value,
            "source": str(paragraph.get("text", "")),
        })

    if not tasks:
        # Remove stale errors for already-successful paragraphs.
        state["errors"] = [
            x for x in state.get("errors", [])
            if isinstance(x, dict) and str(x.get("paragraph_id")) not in translations_by_id
        ]
        state["translations"] = sorted(translations_by_id.values(), key=paragraph_sort_key)
        update_statistics(state, total_paragraphs=len(paragraphs), newly_translated=0)
        _atomic_json(output_path, state)
        print(json.dumps({
            "status": state["status"],
            "message": "No unfinished paragraphs found.",
            "output": str(output_path),
            "statistics": state["statistics"],
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.no_backup:
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)

    # Map existing errors so each failure can be replaced rather than duplicated.
    errors_by_id: dict[str, dict[str, Any]] = {
        str(item.get("paragraph_id")): item
        for item in state.get("errors", [])
        if isinstance(item, dict) and item.get("paragraph_id") is not None
        and str(item.get("paragraph_id")) not in translations_by_id
    }

    factory = ClientFactory(base_url=base_url, timeout_seconds=args.timeout)

    def translate_one(task: dict[str, Any]) -> dict[str, Any]:
        pid_value = task["paragraph_id"]
        source_text = task["source"]
        if not source_text.strip():
            return {
                "ok": True,
                "item": {
                    "paragraph_id": pid_value,
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "translation": "",
                    "model": model,
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "chunk_count": 0,
                    "completed_at": _now(),
                },
            }
        try:
            translation, attempts, latency, chunk_count = _translate_source_text(
                client=factory.get(),
                model=model,
                prompt_profile=prompt_profile,
                source_text=source_text,
                previous_context="",  # Deliberately no document context.
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=args.max_retries,
                source_chunk_chars=source_chunk_chars,
                extra_body=extra_body,
            )
            translation = _clean_translation(translation)
            if not translation:
                raise RuntimeError("model returned an empty translation")
            return {
                "ok": True,
                "item": {
                    "paragraph_id": pid_value,
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "translation": translation,
                    "model": model,
                    "attempts": attempts,
                    "latency_seconds": round(latency, 3),
                    "chunk_count": chunk_count,
                    "zero_english_required": prompt_profile in ZERO_ENGLISH_PROFILES,
                    "latin_residue_policy": "record",
                    "latin_residue_count": len(LATIN_LETTER_RE.findall(translation)),
                    "latin_residue_samples": find_latin_residues(translation),
                    "latin_residue_warning": contains_latin_letters(translation),
                    "completed_at": _now(),
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": {
                    "paragraph_id": pid_value,
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now(),
                },
            }

    newly_translated = 0
    failed_this_run = 0
    submitted = len(tasks)

    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="legal-tc") as pool:
            future_to_task: dict[Future[dict[str, Any]], dict[str, Any]] = {
                pool.submit(translate_one, task): task for task in tasks
            }
            progress = tqdm(
                as_completed(future_to_task),
                total=submitted,
                desc="unfinished paragraphs",
                unit="para",
                dynamic_ncols=True,
            )
            for future in progress:
                task = future_to_task[future]
                pid = str(task["paragraph_id"])
                try:
                    result = future.result()
                except Exception as exc:  # Defensive: translate_one normally captures errors.
                    result = {
                        "ok": False,
                        "error": {
                            "paragraph_id": task["paragraph_id"],
                            "source": task["source"],
                            "source_hash": _text_hash(task["source"]),
                            "error": f"{type(exc).__name__}: {exc}",
                            "time": _now(),
                        },
                    }

                if result["ok"]:
                    item = result["item"]
                    translations_by_id[pid] = item
                    errors_by_id.pop(pid, None)
                    newly_translated += 1
                else:
                    errors_by_id[pid] = result["error"]
                    failed_this_run += 1
                    tqdm.write(f"paragraph {pid} failed: {result['error']['error']}")

                state["translations"] = sorted(translations_by_id.values(), key=paragraph_sort_key)
                state["errors"] = sorted(errors_by_id.values(), key=paragraph_sort_key)
                update_statistics(
                    state,
                    total_paragraphs=len(paragraphs),
                    newly_translated=newly_translated,
                )
                _atomic_json(output_path, state)
                progress.set_postfix(
                    success=newly_translated,
                    failed=failed_this_run,
                    remaining=state["statistics"]["remaining_paragraphs"],
                )

    except KeyboardInterrupt:
        state["translations"] = sorted(translations_by_id.values(), key=paragraph_sort_key)
        state["errors"] = sorted(errors_by_id.values(), key=paragraph_sort_key)
        update_statistics(state, total_paragraphs=len(paragraphs), newly_translated=newly_translated)
        state["status"] = "interrupted"
        _atomic_json(output_path, state)
        raise
    finally:
        factory.close()

    state["translations"] = sorted(translations_by_id.values(), key=paragraph_sort_key)
    state["errors"] = sorted(errors_by_id.values(), key=paragraph_sort_key)
    update_statistics(state, total_paragraphs=len(paragraphs), newly_translated=newly_translated)
    _atomic_json(output_path, state)

    print(json.dumps({
        "status": state["status"],
        "output": str(output_path),
        "dataset": str(dataset_path),
        "workers": args.workers,
        "submitted_unfinished": submitted,
        "newly_translated": newly_translated,
        "failed_this_run": failed_this_run,
        "remaining_errors": len(state.get("errors", [])),
        "statistics": state.get("statistics", {}),
    }, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed results were checkpointed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
