import json
from pathlib import Path

from legal_tc_bench.consistency import (
    aggregate_renderings,
    compute_final_metrics_file,
    normalize_variant,
)


def test_normalize_variant_is_conservative():
    assert normalize_variant(' “生效日期” ') == '生效日期'
    assert normalize_variant('生效日期') != normalize_variant('生效日')


def test_aggregate_renderings(tmp_path: Path):
    rows = [
        {
            "occurrence_id": "d1::T1::0",
            "document_id": "d1",
            "term_id": "T1",
            "source_term": "Effective Date",
            "term_type": "DEFINED_LEGAL_TERM",
            "criticality": "CRITICAL",
            "source_context": "on the Effective Date",
            "target_paragraph": "在生效日期",
            "rendering": "生效日期",
            "extraction_status": "FOUND",
        },
        {
            "occurrence_id": "d1::T1::1",
            "document_id": "d1",
            "term_id": "T1",
            "source_term": "Effective Date",
            "term_type": "DEFINED_LEGAL_TERM",
            "criticality": "CRITICAL",
            "source_context": "after the Effective Date",
            "target_paragraph": "生效日后",
            "rendering": "生效日",
            "extraction_status": "FOUND",
        },
        {
            "occurrence_id": "d1::T1::2",
            "document_id": "d1",
            "term_id": "T1",
            "source_term": "Effective Date",
            "term_type": "DEFINED_LEGAL_TERM",
            "criticality": "CRITICAL",
            "source_context": "Effective Date",
            "target_paragraph": "未出现",
            "rendering": None,
            "extraction_status": "OMITTED",
        },
    ]
    inp = tmp_path / "review.json"
    out = tmp_path / "agg.json"
    inp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summary = aggregate_renderings(inp, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    group = data["groups"][0]
    assert summary["group_count"] == 1
    assert group["variant_count"] == 2
    assert group["total_occurrences"] == 3
    assert group["status_counts"]["OMITTED"] == 1
    assert group["formal_metrics"]["dominant_translation_ratio"] == 0.5


def test_final_metrics_accepts_legal_synonym(tmp_path: Path):
    aggregated = {
        "experiment": "model-a",
        "groups": [{
            "group_id": "d1::T1",
            "document_id": "d1",
            "term_id": "T1",
            "source_term": "Effective Date",
            "term_type": "DEFINED_LEGAL_TERM",
            "criticality": "CRITICAL",
            "total_occurrences": 3,
            "explicit_rendering_occurrences": 3,
            "status_counts": {"FOUND": 3},
            "variant_count": 2,
            "dominant_variant_id": "V000",
            "dominant_rendering": "生效日期",
            "formal_metrics": {
                "dominant_translation_ratio": 0.666667,
                "translation_variation_rate": 0.5,
                "entropy_bits": 0.918296,
                "normalized_entropy": 0.918296,
            },
            "variants": [
                {"variant_id": "V000", "normalized_rendering": "生效日期", "count": 2, "status_counts": {"FOUND": 2}},
                {"variant_id": "V001", "normalized_rendering": "生效日", "count": 1, "status_counts": {"FOUND": 1}},
            ],
        }],
    }
    judgments = {
        "judge_model": "judge",
        "groups": [{
            "group_id": "d1::T1",
            "variant_judgments": [
                {"variant_id": "V000", "label": "EXACT_CONSISTENT", "confidence": 1.0},
                {"variant_id": "V001", "label": "ACCEPTABLE_VARIANT", "confidence": 0.95},
            ],
        }],
    }
    agg = tmp_path / "a.json"
    judge = tmp_path / "j.json"
    out = tmp_path / "m.json"
    agg.write_text(json.dumps(aggregated, ensure_ascii=False), encoding="utf-8")
    judge.write_text(json.dumps(judgments, ensure_ascii=False), encoding="utf-8")
    result = compute_final_metrics_file(agg, judge, out)
    sem = result["semantic_legal_consistency"]
    assert sem["semantic_legal_consistency_rate_micro"] == 1.0
    assert sem["legal_inconsistency_rate_micro"] == 0.0
    assert sem["critical_term_inconsistency_rate"] == 0.0


def test_validate_judge_results_accepts_redundant_dominant_variant():
    from legal_tc_bench.consistency import _validate_judge_results
    batch = [{
        "group_id": "d1::T1",
        "document_id": "d1",
        "term_id": "T1",
        "source_term": "Effective Date",
        "dominant_variant_id": "V000",
        "variants": [
            {"variant_id": "V000", "normalized_rendering": "生效日期"},
            {"variant_id": "V001", "normalized_rendering": "生效日"},
            {"variant_id": "V002", "normalized_rendering": "有效日期"},
        ],
    }]
    parsed = {"results": [{
        "group_id": "d1::T1",
        "confidence": 0.9,
        "rationale": "ok",
        "variant_judgments": [
            {"variant_id": "V000", "label": "ACCEPTABLE_VARIANT", "confidence": 0.9, "rationale": "redundant"},
            {"variant_id": "V001", "label": "ACCEPTABLE_VARIANT", "confidence": 0.9, "rationale": "same"},
            {"variant_id": "V002", "label": "LEGAL_INCONSISTENCY", "confidence": 0.9, "rationale": "different"},
        ],
    }]}
    result = _validate_judge_results(parsed, batch, model="judge")
    assert len(result) == 1
    assert {x["variant_id"] for x in result[0]["variant_judgments"]} == {"V000", "V001", "V002"}


def test_validate_judge_results_reports_missing_variant_ids():
    import pytest
    from legal_tc_bench.consistency import _validate_judge_results
    batch = [{
        "group_id": "d1::T1",
        "document_id": "d1",
        "term_id": "T1",
        "source_term": "Effective Date",
        "dominant_variant_id": "V000",
        "variants": [
            {"variant_id": "V000", "normalized_rendering": "生效日期"},
            {"variant_id": "V001", "normalized_rendering": "生效日"},
            {"variant_id": "V002", "normalized_rendering": "有效日期"},
        ],
    }]
    parsed = {"results": [{
        "group_id": "d1::T1",
        "confidence": 0.9,
        "rationale": "partial",
        "variant_judgments": [
            {"variant_id": "V001", "label": "ACCEPTABLE_VARIANT", "confidence": 0.9, "rationale": "same"},
        ],
    }]}
    with pytest.raises(ValueError, match=r"missing=\['V002'\]"):
        _validate_judge_results(parsed, batch, model="judge")


def test_parse_json_object_reports_empty_response():
    import pytest
    from legal_tc_bench.consistency import EmptyJudgeResponseError, _parse_json_object

    with pytest.raises(EmptyJudgeResponseError, match="Empty model response"):
        _parse_json_object("   ")


def test_parse_json_object_recovers_json_after_preamble():
    from legal_tc_bench.consistency import _parse_json_object

    value = _parse_json_object('analysis text before final answer\n{"results": []}\n')
    assert value == {"results": []}


def test_judge_batch_marks_persistently_empty_variant_unresolved():
    from types import SimpleNamespace

    from legal_tc_bench.consistency import _build_judge_payload, _judge_batch_robust

    class EmptyCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content=None, reasoning_content=None),
                )]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=EmptyCompletions()))
    group = {
        "group_id": "d1::T1",
        "document_id": "d1",
        "term_id": "T1",
        "source_term": "Effective Date",
        "dominant_variant_id": "V000",
        "variants": [
            {"variant_id": "V000", "normalized_rendering": "生效日期", "count": 2},
            {"variant_id": "V001", "normalized_rendering": "有效日期", "count": 1},
        ],
    }
    results, raw, attempts = _judge_batch_robust(
        client, "judge", _build_judge_payload([group]), [group], max_retries=1
    )
    assert attempts >= 2
    assert results[0]["group_label"] == "UNRESOLVED"
    by_id = {x["variant_id"]: x for x in results[0]["variant_judgments"]}
    assert by_id["V001"]["label"] == "UNRESOLVED"
    assert by_id["V001"]["fallback_status"] == "MODEL_OUTPUT_UNUSABLE"
    assert "variant_wise" in raw
