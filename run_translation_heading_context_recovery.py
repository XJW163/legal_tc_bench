#!/usr/bin/env python3
"""Targeted controlled-translation recovery for two Aya heading failures.

The affected source paragraphs are short headings, but Aya-Expanse-32B copied
and expanded the read-only previous context until the completion budget was
exhausted.  This runner preserves the declared experiment settings while
removing previous context only for the two confirmed glossary headings.

Use it as the runner module behind ``run_controlled_contextsafe_v3.py``:

    LEGAL_TC_RUNNER_MODULE=run_translation_heading_context_recovery \
      python3 run_controlled_contextsafe_v3.py ... --methods glossary
"""
from __future__ import annotations

import re
import sys

import legal_tc_bench.controlled_translation as ct
import run_translation_unicode_recovery_v2 as runner


_original_call_controlled_translation = ct._call_controlled_translation
_announced: set[str] = set()

_TARGET_HEADINGS = {
    "ARTICLE 5",
    "SECTION 9.03 Expenses; Limitation of Liability; Indemnity; Etc.",
}


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ").replace("\u202f", " ")).strip()


def _call_controlled_translation_without_heading_context(*args, **kwargs):
    source_text = kwargs.get("source_text")
    constraint_mode = kwargs.get("constraint_mode")
    normalized = (
        _normalize_heading(source_text)
        if isinstance(source_text, str)
        else ""
    )

    if constraint_mode == "glossary" and normalized in _TARGET_HEADINGS:
        retry_kwargs = dict(kwargs)
        retry_kwargs["previous_context"] = ""
        if normalized not in _announced:
            print(
                "heading-context recovery: "
                f"disabled previous context for {normalized!r}",
                file=sys.stderr,
                flush=True,
            )
            _announced.add(normalized)
        return _original_call_controlled_translation(
            *args,
            **retry_kwargs,
        )

    return _original_call_controlled_translation(*args, **kwargs)


ct._call_controlled_translation = (
    _call_controlled_translation_without_heading_context
)

install_request_adapter = runner.install_request_adapter
main = runner.main


if __name__ == "__main__":
    main()
