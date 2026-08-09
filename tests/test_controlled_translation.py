from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_tc_bench import controlled_translation as mod


def test_inject_term_placeholders_longest_and_restore():
    source = "Any Person may sign."
    occurrences = [{
        "term_id": "T0001",
        "source_term": "Person",
        "char_start": 4,
        "char_end": 10,
    }]
    memory = {"T0001": {"canonical_translation": "主体", "legal_scope_summary": "自然人及组织"}}
    protected, constraints = mod.inject_term_placeholders(source, occurrences, memory)
    assert protected == "Any 【术语0001】 may sign."
    assert constraints[0]["canonical_translation"] == "主体"
    mod._validate_placeholder_output("任何【术语0001】均可签署。", constraints)
    assert mod._restore_placeholders("任何【术语0001】均可签署。", constraints) == "任何主体均可签署。"


def test_placeholder_mismatch_rejected():
    constraints = [{"placeholder": "【术语0001】", "canonical_translation": "主体"}]
    with pytest.raises(ValueError, match="placeholder mismatch"):
        mod._validate_placeholder_output("任何主体均可签署。", constraints)


class FakeCompletions:
    def create(self, **kwargs):
        user = kwargs["messages"][-1]["content"]
        if "【术语0001】" in user:
            content = "任何【术语0001】均可签署。"
        else:
            content = "任何主体均可签署。"
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_controlled_batch_translate(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace as SNS
    fake_module = SNS(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    (annotated / "doc.json").write_text(json.dumps({
        "document_id": "doc-1",
        "paragraphs": [{"paragraph_id": 0, "text": "Any Person may sign."}],
        "benchmark_terms": [{
            "term_id": "T0001",
            "source_term": "Person",
            "occurrences": [{"paragraph_id": 0, "char_start": 4, "char_end": 10}],
        }],
    }), encoding="utf-8")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "doc-1.term-memory.json").write_text(json.dumps({
        "document_id": "doc-1",
        "terms": [{
            "term_id": "T0001",
            "source_term": "Person",
            "canonical_translation": "主体",
            "legal_scope_summary": "自然人及组织",
            "criticality": "CRITICAL",
        }],
    }, ensure_ascii=False), encoding="utf-8")

    result = mod.controlled_batch_translate(
        annotated,
        memory,
        tmp_path / "out",
        experiment_name="placeholder-test",
        model="fake",
        base_url="http://fake/v1",
        constraint_mode="placeholder",
    )
    assert result["status"] == "complete"
    output = json.loads((tmp_path / "out" / "placeholder-test" / "doc-1.controlled.translation.json").read_text(encoding="utf-8"))
    assert output["translations"][0]["translation"] == "任何主体均可签署。"
    assert output["translations"][0]["constraint_compliance"] is True


def test_repeated_term_occurrences_receive_unique_placeholders():
    source = "Person may notify another Person."
    occurrences = [
        {"term_id": "T0002", "source_term": "Person", "char_start": 0, "char_end": 6},
        {"term_id": "T0002", "source_term": "Person", "char_start": 26, "char_end": 32},
    ]
    memory = {"T0002": {"canonical_translation": "主体", "legal_scope_summary": "自然人及组织"}}
    protected, constraints = mod.inject_term_placeholders(source, occurrences, memory)
    assert protected == "【术语0002_01】 may notify another 【术语0002_02】."
    assert [x["placeholder"] for x in constraints] == ["【术语0002_01】", "【术语0002_02】"]
    output = "【术语0002_01】可以通知另一【术语0002_02】。"
    mod._validate_placeholder_output(output, constraints)
    assert mod._restore_placeholders(output, constraints) == "主体可以通知另一主体。"


def test_placeholder_diagnostics_distinguish_missing_duplicate_unknown():
    constraints = [
        {"placeholder": "【术语0002_01】", "canonical_translation": "主体"},
        {"placeholder": "【术语0002_02】", "canonical_translation": "主体"},
    ]
    diagnostics = mod._placeholder_diagnostics(
        "【术语0002_01】【术语0002_01】【术语9999】", constraints
    )
    assert diagnostics["missing"] == ["【术语0002_02】"]
    assert diagnostics["duplicated"] == {"【术语0002_01】": 2}
    assert diagnostics["unknown"] == {"【术语9999】": 1}
    assert mod._placeholder_mismatch_score(diagnostics) == 3


class SequenceCompletions:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def create(self, **kwargs):
        content = next(self.outputs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
        )


def test_placeholder_mismatch_is_regenerated_before_fallback():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SequenceCompletions([
        "【术语0002_01】可以签署。",
        "【术语0002_01】可以通知【术语0002_02】。",
    ])))
    constraints = [
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_01】",
            "canonical_translation": "主体",
            "occurrence_index": 1,
        },
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_02】",
            "canonical_translation": "主体",
            "occurrence_index": 2,
        },
    ]
    translation, raw, attempts, _, retries, compliance, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=constraints,
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=3,
        placeholder_semantic_verifier="off",
    )
    assert translation == "主体可以通知主体。"
    assert raw == "【术语0002_01】可以通知【术语0002_02】。"
    assert attempts == 2
    assert retries == 1
    assert compliance is True
    assert meta["strategy"] == "placeholder_retry_exact"
    assert meta["placeholder_regeneration_attempts"] == 1
    assert meta["placeholder_legacy_repair_attempts"] == 0
    assert meta["placeholder_fallback_used"] is False


def test_no_hard_occurrence_is_not_counted_as_placeholder_success():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SequenceCompletions([
        "执行官不得发表贬损性言论。",
    ])))
    translation, raw, attempts, _, retries, compliance, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="The Executive shall not make disparaging remarks.",
        protected_source="The Executive shall not make disparaging remarks.",
        constraints=[],
        soft_constraints=[],
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
    )
    assert translation == raw
    assert attempts == 1
    assert retries == 0
    assert compliance is True
    assert meta["strategy"] == "uncontrolled"


def test_glossary_compliance_accepts_formal_variant_but_not_informal_variant():
    constraints = [{
        "term_id": "T1",
        "source_term": "Base Salary",
        "canonical_translation": "基本工资",
        "allowed_variants": [
            {"rendering": "基本薪资", "condition": "在正式文件中可互换使用"},
            {"rendering": "基薪", "condition": "在非正式或口语化场合使用"},
        ],
    }]
    formal = mod._glossary_compliance_details("基本薪资按月支付。", constraints)
    informal = mod._glossary_compliance_details("基薪按月支付。", constraints)
    assert formal["compliant"] is True
    assert formal["exact_compliant"] is False
    assert formal["terms"][0]["status"] == "APPROVED_VARIANT"
    assert informal["compliant"] is False


class RecordingCompletions:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def create(self, **kwargs):
        self.messages.append(kwargs["messages"])
        content = next(self.outputs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
        )


def _person_constraints():
    return [
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_01】",
            "canonical_translation": "主体",
            "risky_variants": [{"rendering": "个人", "risk_type": "SCOPE_NARROWING"}],
            "occurrence_index": 1,
        },
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_02】",
            "canonical_translation": "主体",
            "risky_variants": [{"rendering": "个人", "risk_type": "SCOPE_NARROWING"}],
            "occurrence_index": 2,
        },
    ]


def test_default_retry_uses_fresh_translation_prompt_not_repair_prompt():
    completions = RecordingCompletions([
        "【术语0002_01】可以签署。",
        "【术语0002_01】可以通知【术语0002_02】。",
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=_person_constraints(),
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        placeholder_semantic_verifier="off",
    )
    assert completions.messages[1][0]["content"] == mod.CONTROLLED_SYSTEM_PROMPT
    assert "需要修复的中文译文" not in completions.messages[1][1]["content"]
    assert "重新完整翻译" in completions.messages[1][1]["content"]


def test_wrapper_contamination_is_rejected_then_regenerated():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SequenceCompletions([
        "个人【术语0002_01】可以通知【术语0002_02】。",
        "【术语0002_01】可以通知【术语0002_02】。",
    ])))
    translation, _, _, _, _, compliance, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=_person_constraints(),
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        placeholder_semantic_verifier="off",
    )
    assert translation == "主体可以通知主体。"
    assert compliance is True
    assert meta["strategy"] == "placeholder_retry_exact"
    assert meta["placeholder_quality_gate_rejections"][0]["stage"] == "context"
    issues = meta["placeholder_quality_gate_rejections"][0]["diagnostics"]["issues"]
    assert issues[0]["issue_type"] == "PLACEHOLDER_WRAPPER_CONTAMINATION"


def test_title_mark_wrapper_and_nested_restore_are_detected():
    constraints = [{
        "term_id": "T0006",
        "source_term": "Code",
        "placeholder": "【术语0006】",
        "canonical_translation": "《国内税收法典》",
    }]
    context = mod._placeholder_context_diagnostics("依据《国内税收法典【术语0006】》第280G条", constraints)
    assert context["error_count"] >= 1
    restored = mod._restored_translation_diagnostics("依据《国内税收法典《国内税收法典》》第280G条", constraints)
    assert restored["error_count"] >= 1
    assert any(x["issue_type"] == "NESTED_TITLE_MARKS" for x in restored["issues"])


def test_retry_semantic_verifier_rejects_added_content_and_falls_back():
    verifier_fail = json.dumps({
        "adequacy": "FAIL",
        "term_semantics": "FAIL",
        "unsupported_addition": True,
        "meaning_omission": False,
        "unnatural_attachment": False,
        "broken_legal_logic": True,
        "problematic_terms": ["Code"],
        "rationale": "新增原文不存在的句子",
    }, ensure_ascii=False)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SequenceCompletions([
        "【术语0002_01】可以签署。",  # inventory failure
        "【术语0002_01】可以通知【术语0002_02】。另有新增事实。",  # retry candidate
        verifier_fail,
        "主体可以通知主体。",  # glossary fallback
    ])))
    translation, _, _, _, _, compliance, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=_person_constraints(),
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        placeholder_semantic_verifier="retry",
    )
    assert translation == "主体可以通知主体。"
    assert compliance is True
    assert meta["strategy"] == "glossary_fallback"
    assert meta["placeholder_semantic_verifier_calls"] == 1
    assert any(x["stage"] == "semantic" for x in meta["placeholder_quality_gate_rejections"])


def test_legacy_llm_repair_remains_explicit_ablation():
    completions = RecordingCompletions([
        "【术语0002_01】可以签署。",
        "【术语0002_01】可以通知【术语0002_02】。",
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    _, _, _, _, _, _, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=_person_constraints(),
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        placeholder_repair_policy="llm_repair",
        placeholder_semantic_verifier="off",
    )
    assert meta["strategy"] == "placeholder_repaired"
    assert meta["placeholder_legacy_repair_attempts"] == 1
    assert completions.messages[1][0]["content"] == mod.PLACEHOLDER_REPAIR_SYSTEM_PROMPT


def test_restored_duplicate_check_is_local_to_inserted_spans():
    constraints = [
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_01】",
            "canonical_translation": "主体",
        },
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_02】",
            "canonical_translation": "主体",
        },
    ]
    translation, spans = mod._restore_placeholders_with_spans(
        "【术语0002_01】可以通知另一【术语0002_02】。",
        constraints,
    )
    diagnostics = mod._restored_translation_diagnostics(
        translation,
        constraints,
        restoration_spans=spans,
    )
    assert translation == "主体可以通知另一主体。"
    assert diagnostics["error_count"] == 0


def test_restored_duplicate_check_rejects_adjacent_duplicate_only():
    constraints = [{
        "term_id": "T0002",
        "source_term": "Person",
        "placeholder": "【术语0002】",
        "canonical_translation": "主体",
    }]
    translation, spans = mod._restore_placeholders_with_spans(
        "主体【术语0002】可以签署。",
        constraints,
    )
    diagnostics = mod._restored_translation_diagnostics(
        translation,
        constraints,
        restoration_spans=spans,
    )
    assert any(
        issue["issue_type"] == "DUPLICATED_CANONICAL_RENDERING"
        for issue in diagnostics["issues"]
    )


def test_acronym_definiendum_wrapper_is_contextually_satisfied():
    constraints = [{
        "term_id": "T0007",
        "source_term": "CEO",
        "placeholder": "【术语0007】",
        "canonical_translation": "首席执行官",
        "usage_status": "DEFINIENDUM",
    }]
    normalized, satisfied, events = mod._normalize_definitional_alias_wrappers(
        "您将向首席执行官（“【术语0007】”）汇报。",
        constraints,
    )
    assert normalized == "您将向首席执行官汇报。"
    assert satisfied == {"【术语0007】"}
    assert events[0]["pattern"] == "CANONICAL_THEN_ALIAS"
    diagnostics = mod._placeholder_diagnostics(
        normalized,
        constraints,
        contextually_satisfied=satisfied,
    )
    assert diagnostics["missing"] == []


def test_sibling_alias_and_explicit_definiendum_collapse_to_one_canonical():
    constraints = [
        {
            "term_id": "T0010",
            "source_term": "Board",
            "placeholder": "【术语0010_01】",
            "canonical_translation": "董事会",
            "usage_status": "DEFINED_TERM_USAGE",
        },
        {
            "term_id": "T0010",
            "source_term": "Board",
            "placeholder": "【术语0010_02】",
            "canonical_translation": "董事会",
            "usage_status": "DEFINIENDUM",
        },
    ]
    normalized, satisfied, _ = mod._normalize_definitional_alias_wrappers(
        "属于【术语0010_01】董事会（即“【术语0010_02】”）的一部分。",
        constraints,
    )
    assert normalized == "属于董事会的一部分。"
    assert satisfied == {"【术语0010_01】", "【术语0010_02】"}


def test_repeated_inventory_failure_falls_back_early():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SequenceCompletions([
        "【术语0002_01】可以签署。",
        "【术语0002_01】可以签署。",
        "主体可以通知主体。",
    ])))
    translation, _, attempts, _, _, compliance, meta = mod._call_controlled_translation(
        client=client,
        model="fake",
        source_text="Person may notify Person.",
        protected_source="【术语0002_01】 may notify 【术语0002_02】.",
        constraints=_person_constraints(),
        previous_context="",
        constraint_mode="placeholder",
        temperature=0.0,
        max_tokens=256,
        max_retries=5,
        placeholder_semantic_verifier="off",
        placeholder_repeat_rejection_limit=2,
    )
    assert translation == "主体可以通知主体。"
    assert compliance is True
    assert attempts == 3
    assert meta["strategy"] == "glossary_fallback"
    assert meta["placeholder_early_fallback"] is True
    assert meta["placeholder_regeneration_attempts"] == 1


def test_term_memory_collision_warning_is_reported_without_rewrite():
    warnings = mod._term_memory_collision_warnings([
        {"term_id": "T1", "source_term": "Cause", "canonical_translation": "正当理由"},
        {"term_id": "T2", "source_term": "Good Reason", "canonical_translation": "正当理由"},
    ])
    assert len(warnings) == 1
    assert warnings[0]["issue_type"] == "CROSS_TERM_CANONICAL_COLLISION"
    assert {x["source_term"] for x in warnings[0]["terms"]} == {"Cause", "Good Reason"}


def test_v172_definition_heading_and_definiendum_are_legitimate_repetitions():
    constraints = [
        {
            "term_id": "T0005",
            "source_term": "Base Salary",
            "placeholder": "【术语0005_01】",
            "canonical_translation": "基本工资",
            "usage_status": "DEFINED_TERM_USAGE",
        },
        {
            "term_id": "T0005",
            "source_term": "Base Salary",
            "placeholder": "【术语0005_02】",
            "canonical_translation": "基本工资",
            "usage_status": "DEFINIENDUM",
        },
    ]
    translation, spans = mod._restore_placeholders_with_spans(
        "(a) 【术语0005_01】。“【术语0005_02】”在本协议第3(a)节中具有所列含义。",
        constraints,
    )
    diagnostics = mod._restored_translation_diagnostics(
        translation,
        constraints,
        restoration_spans=spans,
    )
    assert translation == "(a) 基本工资。“基本工资”在本协议第3(a)节中具有所列含义。"
    assert diagnostics["error_count"] == 0
    assert diagnostics["legitimate_definition_repetition_count"] == 1


def test_v172_title_marked_full_form_alias_is_normalized_and_preserved():
    constraints = [{
        "term_id": "T0019",
        "source_term": "FDIA",
        "placeholder": "【术语0019】",
        "canonical_translation": "联邦存款保险法",
        "usage_status": "DEFINIENDUM",
    }]
    normalized, satisfied, events = mod._normalize_definitional_alias_wrappers(
        "根据《联邦存款保险法》（“【术语0019】”）第8条发出的通知。",
        constraints,
    )
    assert normalized == "根据《联邦存款保险法》第8条发出的通知。"
    assert satisfied == {"【术语0019】"}
    assert events[0]["replacement"] == "《联邦存款保险法》"


def test_v172_alias_normalization_works_at_term_level_with_allowed_variant():
    constraints = [
        {
            "term_id": "T0024",
            "source_term": "Incumbent Board",
            "placeholder": "【术语0024_01】",
            "canonical_translation": "现任董事会",
            "allowed_variants": [{"rendering": "现有董事会"}],
            "usage_status": "DEFINED_TERM_USAGE",
        },
        {
            "term_id": "T0024",
            "source_term": "Incumbent Board",
            "placeholder": "【术语0024_02】",
            "canonical_translation": "现任董事会",
            "allowed_variants": [{"rendering": "现有董事会"}],
            "usage_status": "DEFINIENDUM",
        },
    ]
    normalized, satisfied, events = mod._normalize_definitional_alias_wrappers(
        "经现有董事会（【术语0024_01】）批准，并称为【术语0024_02】。",
        constraints,
    )
    assert normalized == "经现有董事会批准，并称为【术语0024_02】。"
    assert satisfied == {"【术语0024_01】"}
    assert events[0]["rendering_kind"] == "ALLOWED_VARIANT"


def test_v172_placeholder_then_parenthesized_canonical_is_normalized():
    constraints = [
        {
            "term_id": "T0004",
            "source_term": "Compensation Committee",
            "placeholder": "【术语0004_01】",
            "canonical_translation": "薪酬委员会",
            "usage_status": "DEFINED_TERM_USAGE",
        },
        {
            "term_id": "T0004",
            "source_term": "Compensation Committee",
            "placeholder": "【术语0004_02】",
            "canonical_translation": "薪酬委员会",
            "usage_status": "DEFINIENDUM",
        },
    ]
    normalized, satisfied, _ = mod._normalize_definitional_alias_wrappers(
        "由【术语0004_01】（薪酬委员会）批准，并在【术语0004_02】中复核。",
        constraints,
    )
    assert normalized == "由薪酬委员会批准，并在【术语0004_02】中复核。"
    assert satisfied == {"【术语0004_01】"}


def test_v172_two_directly_adjacent_restored_spans_are_rejected():
    constraints = [
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_01】",
            "canonical_translation": "主体",
        },
        {
            "term_id": "T0002",
            "source_term": "Person",
            "placeholder": "【术语0002_02】",
            "canonical_translation": "主体",
        },
    ]
    translation, spans = mod._restore_placeholders_with_spans(
        "【术语0002_01】【术语0002_02】可以签署。",
        constraints,
    )
    diagnostics = mod._restored_translation_diagnostics(
        translation,
        constraints,
        restoration_spans=spans,
    )
    assert any(
        issue["issue_type"] == "ADJACENT_RESTORED_CANONICAL_DUPLICATION"
        for issue in diagnostics["issues"]
    )
