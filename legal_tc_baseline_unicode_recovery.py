#!/usr/bin/env python3
"""Run legal-tc CLI with Unicode space normalization for baseline recovery."""
from __future__ import annotations

import unicodedata

import legal_tc_bench.translate as translate


_original_build_translation_messages = translate._build_translation_messages


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


if __name__ == "__main__":
    from legal_tc_bench.cli import main

    main()
