#!/usr/bin/env python3
"""Run an existing controlled-translation runner with separator passthrough.

Pure underscore/whitespace paragraphs are document layout artifacts, not
translatable language.  They are copied unchanged without making an LLM call.

Set LEGAL_TC_RUNNER_MODULE when the desired runner is not run_translation.py,
for example:
    LEGAL_TC_RUNNER_MODULE=run_translation_unicode_recovery_v2 \
      python run_controlled_with_nonlinguistic_passthrough.py ...
"""
from __future__ import annotations

import importlib
import os
import re
import sys

import legal_tc_bench.controlled_translation as ct


_original_call_controlled_translation = ct._call_controlled_translation
_separator_only = re.compile(r"[\s_*]+", re.UNICODE)


def _call_controlled_translation_with_passthrough(*args, **kwargs):
    source_text = kwargs.get("source_text")
    constraints = kwargs.get("constraints") or []
    soft_constraints = kwargs.get("soft_constraints") or []

    if (
        isinstance(source_text, str)
        and any(character in source_text for character in "_*")
        and _separator_only.fullmatch(source_text)
        and not constraints
        and not soft_constraints
    ):
        details = ct._glossary_compliance_details(source_text, [])
        constraint_mode = kwargs.get("constraint_mode", "glossary")
        control_meta = {
            "strategy": "uncontrolled",
            "placeholder_strategy_version": (
                ct.PLACEHOLDER_STRATEGY_VERSION
                if constraint_mode == "placeholder"
                else None
            ),
            "placeholder_repair_policy": kwargs.get(
                "placeholder_repair_policy", "retry_then_fallback"
            ),
            "placeholder_quality_gate_enabled": bool(
                kwargs.get("placeholder_quality_gate", True)
            ),
            "placeholder_semantic_verifier": kwargs.get(
                "placeholder_semantic_verifier", "retry"
            ),
            "placeholder_regeneration_attempts": 0,
            "placeholder_legacy_repair_attempts": 0,
            "placeholder_quality_gate_rejections": [],
            "placeholder_semantic_verifier_calls": 0,
            "placeholder_early_fallback": False,
            "placeholder_contextual_omission_tokens": [],
            "placeholder_contextual_omission_count": 0,
            "definition_alias_normalizations": [],
            "definition_alias_normalized_occurrence_count": 0,
            "legitimate_definition_repetitions": [],
            "legitimate_definition_repetition_count": 0,
            "max_tokens_escalations": 0,
            "model_request_count": 0,
            "generation_retries": 0,
            "glossary_compliance_retries": 0,
            "placeholder_repair_attempts": 0,
            "placeholder_fallback_used": False,
            "placeholder_diagnostics": None,
            "placeholder_inventory_diagnostics": None,
            "placeholder_context_diagnostics": None,
            "restored_translation_diagnostics": None,
            "placeholder_semantic_verifier_result": None,
            "glossary_compliance_details": details,
            "exact_glossary_compliance": bool(
                details.get("exact_compliant", True)
            ),
            "nonlinguistic_passthrough": True,
        }
        return source_text, source_text, 0, 0.0, 0, True, control_meta

    return _original_call_controlled_translation(*args, **kwargs)


ct._call_controlled_translation = _call_controlled_translation_with_passthrough


_context_token_error = re.compile(
    r"maximum\s*context length is\s+(\d+)\s+tokens.*?"
    r"requested\s+(\d+)\s+output tokens.*?"
    r"contains at least\s+(\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)

_context_character_error = re.compile(
    r"maximum\s*context length is\s+(\d+)\s+tokens.*?"
    r"requested\s+(\d+)\s+output tokens.*?"
    r"prompt contains\s+(\d+)\s+characters",
    re.IGNORECASE | re.DOTALL,
)


def _context_retry_parameters(error_text: str, current_output: int):
    """Return a smaller output limit for either vLLM context-error format."""
    token_match = _context_token_error.search(error_text)
    if token_match:
        context_limit, _requested_output, input_tokens = map(
            int, token_match.groups()
        )
        safe_output = min(
            context_limit - input_tokens - 64,
            current_output - 1024,
        )
        return context_limit, max(256, safe_output), (
            f"input_tokens={input_tokens}"
        )

    character_match = _context_character_error.search(error_text)
    if character_match:
        context_limit, _requested_output, input_characters = map(
            int, character_match.groups()
        )
        # This vLLM error exposes characters rather than token count. Reduce
        # monotonically until the server accepts the largest feasible budget.
        safe_output = max(256, current_output - 1024)
        return context_limit, safe_output, (
            f"input_characters={input_characters}"
        )

    return None


def _install_context_safe_adapter(runner) -> None:
    """Retry an over-budget request with the largest safe output allowance."""
    original_install = runner.install_request_adapter

    def install_request_adapter(*install_args, **install_kwargs):
        original_install(*install_args, **install_kwargs)
        installed_request = ct._request_controlled_completion

        def context_safe_request(*args, **kwargs):
            retry_kwargs = dict(kwargs)
            for _ in range(6):
                try:
                    return installed_request(*args, **retry_kwargs)
                except Exception as exc:
                    current_output = int(retry_kwargs.get("max_tokens", 0))
                    retry = _context_retry_parameters(str(exc), current_output)
                    if retry is None:
                        raise
                    context_limit, safe_output, input_diagnostic = retry
                    if safe_output >= current_output:
                        raise
                    retry_kwargs["max_tokens"] = safe_output
                    print(
                        "context-safe retry: "
                        f"{input_diagnostic}, "
                        f"requested_max_tokens={current_output}, "
                        f"retry_max_tokens={safe_output}, "
                        f"context_limit={context_limit}",
                        file=sys.stderr,
                        flush=True,
                    )
            try:
                return installed_request(*args, **retry_kwargs)
            except Exception as exc:
                current_output = int(retry_kwargs.get("max_tokens", 0))
                if _context_retry_parameters(str(exc), current_output):
                    raise RuntimeError(
                        "unable to fit request within the model context window "
                        "after six safe reductions"
                    ) from exc
                raise

        ct._request_controlled_completion = context_safe_request

    runner.install_request_adapter = install_request_adapter


if __name__ == "__main__":
    runner_module = os.environ.get("LEGAL_TC_RUNNER_MODULE", "run_translation")
    runner = importlib.import_module(runner_module)
    _install_context_safe_adapter(runner)
    runner.main()
