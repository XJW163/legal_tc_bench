#!/usr/bin/env python3
"""Resume failed baseline translations under a tight model context window.

This is a recovery wrapper around the existing ``legal-tc`` CLI.  It keeps the
experiment settings passed to the CLI unchanged.  Only after a paragraph has
failed because of context overflow or output truncation does it retry without
read-only previous-paragraph context and, if necessary, split the source into
progressively smaller chunks.

Unicode separator spaces are normalized in prompts to avoid pathological
generation on U+202F and related spacing characters.
"""
from __future__ import annotations

import os
import sys
import unicodedata
from typing import Any

import legal_tc_bench.translate as translate


_original_build_translation_messages = translate._build_translation_messages
_original_translate_source_text = translate._translate_source_text


def _normalize_unicode_spaces(value: str) -> str:
    return "".join(
        " " if character != " " and unicodedata.category(character) == "Zs"
        else character
        for character in value
    )


def _build_normalized_translation_messages(
    *,
    prompt_profile: str,
    source_text: str,
    previous_context: str,
):
    return _original_build_translation_messages(
        prompt_profile=prompt_profile,
        source_text=_normalize_unicode_spaces(source_text),
        previous_context=_normalize_unicode_spaces(previous_context),
    )


translate._build_translation_messages = _build_normalized_translation_messages


def _install_request_endpoint_override() -> None:
    """Redirect network calls while preserving the recorded experiment URL.

    Set ``LEGAL_TC_RECOVERY_REQUEST_BASE_URL`` to the currently reachable
    endpoint and pass the original manifest base URL to the CLI.  This keeps
    existing experiment signatures stable while moving only the transport.
    """
    request_base_url = os.environ.get(
        "LEGAL_TC_RECOVERY_REQUEST_BASE_URL", ""
    ).strip()
    if not request_base_url:
        return

    import openai

    original_openai = openai.OpenAI
    announced = False

    def redirected_openai(*args, **kwargs):
        nonlocal announced
        recorded_base_url = kwargs.get("base_url")
        kwargs["base_url"] = request_base_url
        if not announced:
            print(
                "baseline recovery endpoint override: "
                f"recorded_base_url={recorded_base_url}, "
                f"request_base_url={request_base_url}",
                file=sys.stderr,
                flush=True,
            )
            announced = True
        return original_openai(*args, **kwargs)

    openai.OpenAI = redirected_openai


def _is_recoverable_failure(exc: Exception) -> bool:
    message = str(exc).lower().replace("maximumcontext", "maximum context")
    return any(
        marker in message
        for marker in (
            "maximum context length",
            "context window",
            "truncated by max_tokens",
            "truncated by max tokens",
            "translation was truncated",
        )
    )


def _recovery_chunk_limits(source_length: int, configured_limit: int) -> list[int]:
    candidates = [configured_limit, 6000, 4000, 2500, 1500, 900, 500, 250]
    limits: list[int] = []
    for candidate in candidates:
        limit = max(1, int(candidate))
        if limit not in limits:
            limits.append(limit)
    # A limit larger than the source is still useful for the first no-context
    # attempt.  Later duplicate one-chunk attempts are skipped.
    return limits


def _translate_source_text_contextsafe(
    *,
    client: Any,
    model: str,
    prompt_profile: str,
    source_text: str,
    previous_context: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    source_chunk_chars: int,
    extra_body: dict[str, Any] | None,
) -> tuple[str, int, float, int]:
    """Recover one missing paragraph without changing its manifest signature."""
    last_error: Exception | None = None
    attempted_shapes: set[tuple[int, ...]] = set()

    for chunk_limit in _recovery_chunk_limits(
        len(source_text), source_chunk_chars
    ):
        pieces = translate._split_source_text(source_text, chunk_limit)
        piece_count = len(pieces)
        piece_shape = tuple(len(piece) for piece in pieces)
        if piece_shape in attempted_shapes:
            continue
        attempted_shapes.add(piece_shape)

        print(
            "baseline context-safe recovery: "
            f"source_characters={len(source_text)}, "
            f"chunk_limit={chunk_limit}, chunks={piece_count}, "
            "previous_context=disabled",
            file=sys.stderr,
            flush=True,
        )

        try:
            # max_retries=1 prevents the baseline implementation from doubling
            # 4096 to 8192 after a truncation, which would leave no input budget
            # on an 8192-token endpoint.  A truncation instead advances to the
            # next, smaller source-chunk plan.
            return _original_translate_source_text(
                client=client,
                model=model,
                prompt_profile=prompt_profile,
                source_text=source_text,
                previous_context="",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                source_chunk_chars=chunk_limit,
                extra_body=extra_body,
            )
        except Exception as exc:
            if not _is_recoverable_failure(exc):
                raise
            last_error = exc
            print(
                "baseline context-safe retry: "
                f"chunk_limit={chunk_limit}, error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if last_error is not None:
        raise RuntimeError(
            "baseline recovery failed after all context-safe chunk plans"
        ) from last_error
    raise RuntimeError("baseline recovery produced no retry plan")


translate._translate_source_text = _translate_source_text_contextsafe


if __name__ == "__main__":
    _install_request_endpoint_override()
    from legal_tc_bench.cli import main

    main()
