#!/usr/bin/env python3
"""Run final controlled translation for one currently loaded vLLM model.

Examples:
  python run_translation.py \
    --model gpt-oss-20b \
    --base-url http://10.12.143.51:8001/v1 \
    --model-key gpt-oss-20b \
    --methods glossary placeholder

  python run_translation.py \
    --model Qwen3-8B \
    --base-url http://10.12.143.51:8000/v1 \
    --model-key qwen3-8b \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_json_object(value: str | None, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return parsed


def advertised_models(base_url: str, timeout: float = 10.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot query {url}: {exc}") from exc
    return [str(x.get("id")) for x in payload.get("data", []) if x.get("id")]


def install_request_adapter(
    extra_body: dict[str, Any],
    request_parameters: dict[str, Any],
    normalize_unicode_spaces: bool = False,
) -> None:
    """Inject optional model-specific OpenAI request fields."""
    import legal_tc_bench.controlled_translation as ct

    if not extra_body and not request_parameters and not normalize_unicode_spaces:
        return

    def request_controlled_completion(
        *,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        request_messages = messages
        if normalize_unicode_spaces:
            request_messages = [
                {
                    **message,
                    "content": re.sub(
                        r"[\u00a0\u202f]+",
                        " ",
                        str(message.get("content", "")),
                    ),
                }
                for message in messages
            ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        kwargs.update(request_parameters)
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw_output = choice.message.content or ""
        if getattr(choice, "finish_reason", None) == "length":
            cleaned_output = ct._clean_translation(raw_output)
            reasoning_output = getattr(choice.message, "reasoning_content", None) or ""
            usage = getattr(response, "usage", None)
            diagnostics = json.dumps(
                {
                    "requested_max_tokens": max_tokens,
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "raw_output_characters": len(raw_output),
                    "cleaned_output_characters": len(cleaned_output),
                    "reasoning_output_characters": len(reasoning_output),
                    "raw_beginning": raw_output[:500],
                    "raw_ending": raw_output[-800:],
                    "cleaned_beginning": cleaned_output[:300],
                    "cleaned_ending": cleaned_output[-600:],
                    "reasoning_beginning": reasoning_output[:300],
                    "reasoning_ending": reasoning_output[-600:],
                },
                ensure_ascii=False,
            )
            raise ct.ControlledTranslationTruncatedError(
                f"controlled translation truncated by max_tokens: {diagnostics}"
            )
        output = ct._clean_translation(raw_output)
        if not output:
            raise ValueError("model returned empty translation")
        return output

    ct._request_controlled_completion = request_controlled_completion


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run glossary and/or closed-loop placeholder translation for one loaded model."
    )
    parser.add_argument("--model", required=True, help="Model ID advertised by /v1/models")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL ending in /v1")
    parser.add_argument("--model-key", required=True, help="Stable filesystem name, e.g. gpt-oss-20b")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["glossary", "placeholder"],
        default=["glossary", "placeholder"],
    )
    parser.add_argument("--annotated", default="data/annotated")
    parser.add_argument("--term-memory", default="data/term_memory_frozen_v1")
    parser.add_argument("--occurrence-validation", default="data/occurrences_validated_v161")
    parser.add_argument("--output-root", default="data/translations_controlled")
    parser.add_argument("--experiment-suffix", default="final")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-paragraphs", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--document-ids", help="Comma-separated exact document IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument(
        "--extra-body",
        help='JSON object, e.g. {"chat_template_kwargs":{"enable_thinking":false}}',
    )
    parser.add_argument(
        "--request-parameters",
        help='Extra top-level request fields as JSON, e.g. {"reasoning_effort":"low"}',
    )
    parser.add_argument(
        "--normalize-unicode-spaces",
        action="store_true",
        help="Collapse runs of U+00A0/U+202F to one ASCII space in model requests only.",
    )
    parser.add_argument("--placeholder-repair-policy", default="retry_then_fallback")
    parser.add_argument("--placeholder-semantic-verifier", default="retry")
    parser.add_argument("--placeholder-repeat-rejection-limit", type=int, default=2)
    parser.add_argument("--semantic-verifier-retries", type=int, default=2)
    args = parser.parse_args()

    annotated = Path(args.annotated)
    term_memory = Path(args.term_memory)
    occurrence_validation = Path(args.occurrence_validation)
    for path, label in [
        (annotated, "annotated data"),
        (term_memory, "term memory"),
        (occurrence_validation, "occurrence validation"),
    ]:
        if not path.is_dir():
            raise SystemExit(f"missing {label}: {path}")

    if not args.skip_server_check:
        available = advertised_models(args.base_url)
        print("advertised models:", available)
        if args.model not in available:
            raise SystemExit(
                f"endpoint advertises {available}, not {args.model!r}; "
                "load the requested model or pass the exact advertised ID"
            )

    extra_body = parse_json_object(args.extra_body, "--extra-body")
    request_parameters = parse_json_object(args.request_parameters, "--request-parameters")
    install_request_adapter(
        extra_body,
        request_parameters,
        normalize_unicode_spaces=args.normalize_unicode_spaces,
    )

    from legal_tc_bench.controlled_translation import controlled_batch_translate

    output_root = Path(args.output_root)
    document_ids = (
        {x.strip() for x in args.document_ids.split(",") if x.strip()}
        if args.document_ids
        else None
    )
    summaries: list[dict[str, Any]] = []

    for method in args.methods:
        experiment = f"{args.model_key}-{method}-{args.experiment_suffix}"
        print(f"\n=== {experiment} ===", flush=True)
        result = controlled_batch_translate(
            annotated,
            term_memory,
            output_root,
            experiment_name=experiment,
            model=args.model,
            base_url=args.base_url,
            constraint_mode=method,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_paragraphs=args.context_paragraphs,
            max_context_chars=args.max_context_chars,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            workers=args.workers,
            overwrite=args.overwrite,
            continue_on_error=not args.fail_fast,
            max_documents=args.max_documents,
            document_ids=document_ids,
            occurrence_validation=occurrence_validation,
            placeholder_repair_policy=args.placeholder_repair_policy,
            placeholder_quality_gate=True,
            placeholder_semantic_verifier=(
                args.placeholder_semantic_verifier if method == "placeholder" else "off"
            ),
            placeholder_repeat_rejection_limit=args.placeholder_repeat_rejection_limit,
            semantic_verifier_retries=args.semantic_verifier_retries,
        )
        summaries.append({"method": method, "experiment": experiment, **result})
        print(json.dumps(result, ensure_ascii=False, indent=2))

    summary_path = output_root / f"{args.model_key}-{args.experiment_suffix}.run-summary.json"
    atomic_json(
        summary_path,
        {
            "model_key": args.model_key,
            "model": args.model,
            "base_url": args.base_url,
            "request_extra_body": extra_body,
            "request_parameters": request_parameters,
            "normalize_unicode_spaces": args.normalize_unicode_spaces,
            "runs": summaries,
        },
    )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
