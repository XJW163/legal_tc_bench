from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from legal_tc_bench import translation_evaluation as mod


VALID_JUDGMENT = {
    "scores": {
        "A": {
            "semantic_fidelity": 4,
            "legal_accuracy": 4,
            "terminology_compliance": 3,
            "fluency": 4,
            "overall_quality": 4,
        },
        "B": {
            "semantic_fidelity": 5,
            "legal_accuracy": 5,
            "terminology_compliance": 5,
            "fluency": 4,
            "overall_quality": 5,
        },
    },
    "critical_errors": {
        "A": {
            "meaning_omission": False,
            "unsupported_addition": False,
            "polarity_or_modality_error": False,
            "party_or_role_error": False,
            "number_date_or_reference_error": False,
        },
        "B": {
            "meaning_omission": False,
            "unsupported_addition": False,
            "polarity_or_modality_error": False,
            "party_or_role_error": False,
            "number_date_or_reference_error": False,
        },
    },
    "winner": "B",
    "confidence": 0.9,
    "rationale": "受控译文术语更准确，法律含义保持完整。",
}


class FakeCompletions:
    calls: list[dict] = []
    outputs: list[str] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.outputs.pop(0) if self.outputs else json.dumps(
            VALID_JUDGMENT, ensure_ascii=False
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    annotated = root / "annotated"
    baseline = root / "baseline"
    controlled = root / "controlled"
    memory = root / "memory"
    for path in (annotated, baseline, controlled, memory):
        path.mkdir()

    (annotated / "doc.json").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "paragraphs": [
                    {"paragraph_id": 0, "text": "The Company shall pay."},
                    {"paragraph_id": 1, "text": "Cause means fraud."},
                ],
                "benchmark_terms": [
                    {
                        "term_id": "T1",
                        "source_term": "Cause",
                        "annotation": {"criticality": "CRITICAL"},
                        "occurrences": [
                            {"paragraph_id": 1, "char_start": 0, "char_end": 5}
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (baseline / "doc.translation.json").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "translations": [
                    {"paragraph_id": 0, "translation": "公司支付。"},
                    {"paragraph_id": 1, "translation": "正当理由指欺诈。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (controlled / "doc.controlled.translation.json").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "translations": [
                    {"paragraph_id": 0, "translation": "公司应支付。"},
                    {"paragraph_id": 1, "translation": "因故是指欺诈。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (memory / "doc-1.term-memory.json").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "terms": [
                    {
                        "term_id": "T1",
                        "source_term": "Cause",
                        "canonical_translation": "因故",
                        "legal_scope_summary": "重大过失、欺诈等终止事由",
                        "criticality": "CRITICAL",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return annotated, baseline, controlled, memory


def test_evaluate_translation_tree_and_resume(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    FakeCompletions.calls = []
    FakeCompletions.outputs = []
    annotated, baseline, controlled, memory = _write_inputs(tmp_path)

    result = mod.evaluate_translation_tree(
        annotated,
        baseline,
        controlled,
        tmp_path / "evaluations",
        experiment_name="deepseek-pilot",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        term_memory=memory,
        position_control="fixed",
        workers=1,
        evaluation_scope="all",
    )
    assert result["status"] == "complete"
    assert result["paragraphs_evaluated"] == 2
    assert len(FakeCompletions.calls) == 2

    exp = tmp_path / "evaluations" / "deepseek-pilot"
    output = json.loads(
        (exp / "doc-1.translation-evaluation.json").read_text(encoding="utf-8")
    )
    assert output["evaluations"][0]["judgment"]["winner"] == "controlled"
    assert output["evaluations"][1]["relevant_terms"][0]["canonical_translation"] == "因故"

    summary = json.loads(
        (exp / "translation_evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["overall"]["wins"]["controlled"] == 2
    assert summary["usage"]["total_tokens"] == 300

    second = mod.evaluate_translation_tree(
        annotated,
        baseline,
        controlled,
        tmp_path / "evaluations",
        experiment_name="deepseek-pilot",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        term_memory=memory,
        position_control="fixed",
        workers=1,
        evaluation_scope="all",
    )
    assert second["status"] == "complete"
    assert len(FakeCompletions.calls) == 2


def test_malformed_json_is_retried(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    FakeCompletions.calls = []
    FakeCompletions.outputs = [
        "not json",
        "```json\n" + json.dumps(VALID_JUDGMENT, ensure_ascii=False) + "\n```",
    ]
    annotated, baseline, controlled, memory = _write_inputs(tmp_path)

    result = mod.evaluate_translation_tree(
        annotated,
        baseline,
        controlled,
        tmp_path / "evaluations",
        experiment_name="retry-pilot",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        term_memory=memory,
        position_control="fixed",
        max_documents=1,
        max_paragraphs_per_document=1,
        max_retries=2,
        workers=1,
        evaluation_scope="all",
    )
    assert result["status"] == "complete"
    assert len(FakeCompletions.calls) == 2


def test_bidirectional_disagreement_becomes_tie():
    first = mod._map_blind_judgment(
        mod._validate_blind_judgment(VALID_JUDGMENT),
        role_a="baseline",
        role_b="controlled",
    )
    reversed_value = json.loads(json.dumps(VALID_JUDGMENT))
    reversed_value["winner"] = "B"
    second = mod._map_blind_judgment(
        mod._validate_blind_judgment(reversed_value),
        role_a="controlled",
        role_b="baseline",
    )
    combined = mod._combine_bidirectional_judgments([first, second])
    assert first["winner"] == "controlled"
    assert second["winner"] == "baseline"
    assert combined["winner"] == "tie"
    assert combined["position_disagreement"] is True


def test_single_paragraph_failure_does_not_abort_document(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    FakeCompletions.calls = []
    FakeCompletions.outputs = [
        "not json",
        json.dumps(VALID_JUDGMENT, ensure_ascii=False),
    ]
    annotated, baseline, controlled, memory = _write_inputs(tmp_path)

    result = mod.evaluate_translation_tree(
        annotated,
        baseline,
        controlled,
        tmp_path / "evaluations",
        experiment_name="fault-tolerant-pilot",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        term_memory=memory,
        position_control="fixed",
        max_retries=1,
        workers=1,
        evaluation_scope="all",
    )
    assert result["status"] == "complete_with_failures"
    assert result["paragraphs_evaluated"] == 1
    assert result["paragraphs_failed"] == 1
    assert len(FakeCompletions.calls) == 2



def test_low_cost_scope_keeps_terms_and_skips_non_terms(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    FakeCompletions.calls = []
    FakeCompletions.outputs = []
    annotated, baseline, controlled, memory = _write_inputs(tmp_path)

    result = mod.evaluate_translation_tree(
        annotated,
        baseline,
        controlled,
        tmp_path / "evaluations",
        experiment_name="low-cost-pilot",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        term_memory=memory,
        position_control="fixed",
        workers=1,
        evaluation_scope="term-bearing-plus-sample",
        non_term_sample_rate=0.0,
    )
    assert result["status"] == "complete"
    assert result["paragraphs_evaluated"] == 1
    assert len(FakeCompletions.calls) == 1
    assert FakeCompletions.calls[0]["response_format"] == {"type": "json_object"}

    output = json.loads(
        (
            tmp_path
            / "evaluations"
            / "low-cost-pilot"
            / "doc-1.translation-evaluation.json"
        ).read_text(encoding="utf-8")
    )
    assert output["evaluations"][0]["paragraph_id"] == 1
    assert output["selection"]["candidate_paragraphs"] == 2
    assert output["selection"]["selected_paragraphs"] == 1
    assert output["selection"]["selected_term_bearing_paragraphs"] == 1
    assert output["selection"]["selected_non_term_paragraphs"] == 0


def test_deterministic_sampling_is_reproducible():
    first = [
        mod._deterministic_sample_hit(
            document_id="doc",
            paragraph_id=str(pid),
            rate=0.25,
            seed=2026,
        )
        for pid in range(100)
    ]
    second = [
        mod._deterministic_sample_hit(
            document_id="doc",
            paragraph_id=str(pid),
            rate=0.25,
            seed=2026,
        )
        for pid in range(100)
    ]
    assert first == second
    assert 10 <= sum(first) <= 40


class TruncatingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="{\"scores\":"),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=600,
                    total_tokens=700,
                ),
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(VALID_JUDGMENT, ensure_ascii=False)
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


def test_truncation_escalates_once_instead_of_repeating_same_limit():
    completions = TruncatingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    parsed, _, attempts, usage = mod._call_judge_with_retry(
        client=client,
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "return JSON"}],
        temperature=0.0,
        max_tokens=600,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
        json_mode=True,
        max_token_escalations=1,
    )
    assert parsed["winner"] == "B"
    assert attempts == 2
    assert [call["max_tokens"] for call in completions.calls] == [600, 1200]
    assert all(
        call["response_format"] == {"type": "json_object"}
        for call in completions.calls
    )
    assert usage["total_tokens"] == 850


def test_truncation_without_escalation_stops_immediately():
    completions = TruncatingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        mod._call_judge_with_retry(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "return JSON"}],
            temperature=0.0,
            max_tokens=600,
            max_retries=5,
            extra_body=None,
            json_mode=True,
            max_token_escalations=0,
        )
    except mod.TranslationEvaluationError as exc:
        assert "truncated" in str(exc)
    else:
        raise AssertionError("expected TranslationEvaluationError")
    assert len(completions.calls) == 1
