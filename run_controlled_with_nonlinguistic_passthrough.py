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

import legal_tc_bench.controlled_translation as ct


_original_call_controlled_translation = ct._call_controlled_translation
_separator_only = re.compile(r"[\s_]+", re.UNICODE)


def _call_controlled_translation_with_passthrough(*args, **kwargs):
    source_text = kwargs.get("source_text")
    constraints = kwargs.get("constraints") or []
    soft_constraints = kwargs.get("soft_constraints") or []

    if (
        isinstance(source_text, str)
        and "_" in source_text
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


if __name__ == "__main__":
    runner_module = os.environ.get("LEGAL_TC_RUNNER_MODULE", "run_translation")
    runner = importlib.import_module(runner_module)
    runner.main()

