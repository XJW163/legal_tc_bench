from __future__ import annotations

import json
from pathlib import Path

from legal_tc_bench.occurrence_validation import (
    occurrence_map_by_paragraph,
    validate_document_occurrences,
    validate_occurrence,
    validate_term_occurrence_tree,
)
from legal_tc_bench.controlled_translation import prepare_term_controls


def _occ(text: str, token: str, occurrence_number: int = 1) -> dict:
    start = -1
    cursor = 0
    for _ in range(occurrence_number):
        start = text.index(token, cursor)
        cursor = start + len(token)
    return {"paragraph_id": 0, "char_start": start, "char_end": start + len(token)}


def test_lowercase_cause_is_excluded_as_common_word():
    text = "The Executive shall not make, or cause to be made, any statement."
    result = validate_occurrence(
        text,
        source_term="Cause",
        term_id="T0003",
        occurrence=_occ(text, "cause"),
        is_definition_paragraph=False,
    )
    assert result["usage_status"] == "COMMON_WORD_USAGE"
    assert result["control_eligibility"] == "NONE"
    assert result["validation_reasons"] == ["CASE_MISMATCH_WITH_DEFINED_TERM"]


def test_code_definition_and_other_legal_name_are_distinguished():
    text = '(d) Code. “Code” shall mean the Internal Revenue Code of 1986, as amended.'
    first = validate_occurrence(
        text, source_term="Code", term_id="T0006", occurrence=_occ(text, "Code", 1),
        is_definition_paragraph=True,
    )
    second = validate_occurrence(
        text, source_term="Code", term_id="T0006", occurrence=_occ(text, "Code", 2),
        is_definition_paragraph=True,
    )
    third = validate_occurrence(
        text, source_term="Code", term_id="T0006", occurrence=_occ(text, "Code", 3),
        is_definition_paragraph=True,
    )
    assert first["control_eligibility"] == "HARD"
    assert second["usage_status"] == "DEFINIENDUM"
    assert second["control_eligibility"] == "HARD"
    assert third["usage_status"] == "OTHER_LEGAL_NAME_COMPONENT"
    assert third["definition_role"] == "DEFINIENS_COMPONENT"
    assert third["control_eligibility"] == "NONE"
    assert third["compound_context"] == "Internal Revenue Code"


def test_united_states_code_component_is_excluded():
    text = "Sections of Title 18 of the United States Code apply."
    result = validate_occurrence(
        text,
        source_term="Code",
        term_id="T0006",
        occurrence=_occ(text, "Code"),
        is_definition_paragraph=False,
    )
    assert result["control_eligibility"] == "NONE"
    assert result["compound_context"] == "United States Code"


def test_capitalised_modifier_is_soft_not_placeholder():
    text = "Any Sanctioned Person may be blocked."
    result = validate_occurrence(
        text,
        source_term="Person",
        term_id="T0002",
        occurrence=_occ(text, "Person"),
        is_definition_paragraph=False,
    )
    assert result["control_eligibility"] == "SOFT"
    assert result["compound_context"] == "Sanctioned Person"

    protected, hard, soft, excluded = prepare_term_controls(
        text,
        [result],
        {"T0002": {"canonical_translation": "主体", "legal_scope_summary": "自然人及组织"}},
    )
    assert protected == text
    assert hard == []
    assert len(soft) == 1
    assert excluded == []


def test_validate_tree_writes_reusable_document_artifact(tmp_path: Path):
    text = '(d) Code. “Code” shall mean the Internal Revenue Code of 1986, as amended.'
    starts = []
    cursor = 0
    while True:
        start = text.find("Code", cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 4
    document = {
        "document_id": "doc-1",
        "paragraphs": [{"paragraph_id": 0, "text": text}],
        "benchmark_terms": [{
            "term_id": "T0006",
            "source_term": "Code",
            "definition_paragraph_ids": [0],
            "occurrences": [
                {"paragraph_id": 0, "char_start": start, "char_end": start + 4}
                for start in starts
            ],
        }],
    }
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    (annotated / "doc.json").write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "validated"
    manifest = validate_term_occurrence_tree(annotated, output)
    assert manifest["totals"]["hard_occurrences"] == 2
    assert manifest["totals"]["excluded_occurrences"] == 1
    artifact = json.loads((output / "doc-1.occurrence-validation.json").read_text(encoding="utf-8"))
    by_pid = occurrence_map_by_paragraph(artifact)
    assert len(by_pid["0"]) == 3
    assert [x["control_eligibility"] for x in by_pid["0"]] == ["HARD", "HARD", "NONE"]


def test_validate_document_occurrences_counts_hard_soft_none():
    paragraphs = [
        "Any Person may act.",
        "Any Sanctioned Person may act.",
        "No person may act.",
    ]
    document = {
        "document_id": "doc",
        "paragraphs": [{"paragraph_id": i, "text": text} for i, text in enumerate(paragraphs)],
        "benchmark_terms": [{
            "term_id": "T0001",
            "source_term": "Person",
            "definition_paragraph_ids": [],
            "occurrences": [
                {"paragraph_id": 0, "char_start": 4, "char_end": 10},
                {"paragraph_id": 1, "char_start": 15, "char_end": 21},
                {"paragraph_id": 2, "char_start": 3, "char_end": 9},
            ],
        }],
    }
    result = validate_document_occurrences(document)
    stats = result["statistics"]
    assert stats["hard_occurrences"] == 1
    assert stats["soft_occurrences"] == 1
    assert stats["excluded_occurrences"] == 1


def _validate(text: str, source_term: str, token: str | None = None, occurrence_number: int = 1):
    token = token or source_term
    return validate_occurrence(
        text,
        source_term=source_term,
        term_id="T",
        occurrence=_occ(text, token, occurrence_number),
        is_definition_paragraph=False,
    )


def test_sentence_boundary_does_not_create_false_compound():
    result = _validate("Benefits. Gilead will provide coverage.", "Gilead")
    assert result["control_eligibility"] == "HARD"
    assert result["usage_status"] == "DEFINED_TERM_USAGE"
    assert result["compound_context"] is None


def test_external_possessive_and_determiner_keep_core_term_hard():
    possessive = _validate("The Executive’s Base Salary shall continue.", "Base Salary")
    assert possessive["control_eligibility"] == "HARD"
    assert possessive["usage_status"] == "EXTERNAL_MODIFIED_DEFINED_TERM"
    assert possessive["validation_reasons"] == ["POSSESSIVE_MODIFIER_OUTSIDE_TERM"]

    determiner = _validate("Your Sign-On Bonus will be paid.", "Sign-On Bonus")
    assert determiner["control_eligibility"] == "HARD"
    assert determiner["validation_reasons"] == ["DETERMINER_OR_POSSESSIVE_MODIFIER"]


def test_safe_modifier_for_acronym_keeps_term_hard():
    annual = _validate("The Annual LTIP is subject to approval.", "LTIP")
    existing = _validate("The Existing LTIP remains effective.", "LTIP")
    assert annual["control_eligibility"] == "HARD"
    assert existing["control_eligibility"] == "HARD"
    assert annual["validation_reasons"] == ["SAFE_EXTERNAL_MODIFIER"]
    assert existing["validation_reasons"] == ["SAFE_EXTERNAL_MODIFIER"]


def test_job_title_component_is_excluded():
    chief = _validate("The Chief Executive Officer approved it.", "Executive")
    vice = _validate("The Executive Vice President approved it.", "Executive")
    assert chief["control_eligibility"] == "NONE"
    assert vice["control_eligibility"] == "NONE"
    assert chief["validation_reasons"] == ["COMPONENT_OF_JOB_TITLE"]
    assert vice["context_classification"] == "JOB_TITLE"


def test_legal_instrument_component_is_excluded():
    result = _validate("The Defend Trade Secrets Act applies.", "Trade Secrets")
    assert result["control_eligibility"] == "NONE"
    assert result["usage_status"] == "OTHER_NAMED_EXPRESSION_COMPONENT"
    assert result["validation_reasons"] == ["COMPONENT_OF_LEGAL_INSTRUMENT_NAME"]
    assert result["compound_context"] == "Defend Trade Secrets Act"


def test_entity_full_name_component_is_excluded_but_relation_usage_is_hard():
    full_name = _validate("Gilead Sciences issued the notice.", "Gilead")
    relation = _validate("Gilead Leadership approved the change.", "Gilead")
    tanger = _validate("Tanger Team will review the award.", "Tanger")
    bank = _validate("U.S. Century Bank is the employer.", "Bank")

    assert full_name["control_eligibility"] == "NONE"
    assert full_name["validation_reasons"] == ["COMPONENT_OF_ENTITY_FULL_NAME"]
    assert relation["control_eligibility"] == "HARD"
    assert relation["usage_status"] == "ENTITY_RELATION_USAGE"
    assert tanger["control_eligibility"] == "HARD"
    assert bank["control_eligibility"] == "NONE"


def test_generic_named_head_is_excluded_but_possessive_head_is_hard():
    savings = _validate("The Savings Plan remains in force.", "Plan")
    board = _validate("Gilead’s Board approved it.", "Board")
    assert savings["control_eligibility"] == "NONE"
    assert savings["validation_reasons"] == ["COMPONENT_OF_NAMED_EXPRESSION"]
    assert board["control_eligibility"] == "HARD"
    assert board["validation_reasons"] == ["POSSESSIVE_MODIFIER_OUTSIDE_TERM"]


def test_ambiguous_semantic_modifier_remains_soft():
    result = _validate("Any Sanctioned Person may be blocked.", "Person")
    assert result["control_eligibility"] == "SOFT"
    assert result["context_classification"] == "AMBIGUOUS_CAPITALISED_COMPOUND"


def test_all_caps_heading_before_entity_alias_is_hard():
    same_line = _validate("POLICY Tanger employees must comply.", "Tanger")
    newline = _validate("POLICY\nTanger employees must comply.", "Tanger")
    for result in (same_line, newline):
        assert result["control_eligibility"] == "HARD"
        assert result["usage_status"] == "DEFINED_TERM_USAGE"
        assert result["validation_reasons"] == ["ALL_CAPS_HEADING_BOUNDARY"]
        assert result["context_classification"] == "HEADING_BOUNDARY"
