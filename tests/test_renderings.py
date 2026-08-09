import json
from pathlib import Path

from legal_tc_bench.evaluate import locate_term_translations
from legal_tc_bench.renderings import _parse_json, _validate_results


def test_make_review_uses_benchmark_terms(tmp_path: Path):
    dataset = {
        "document_id": "doc1",
        "terms": [{
            "term_id": "T9999",
            "source_term": "Excluded",
            "occurrences": [{"paragraph_id": 0, "context": "Excluded"}],
        }],
        "benchmark_terms": [{
            "term_id": "T0001",
            "source_term": "Effective Date",
            "annotation": {
                "term_type": "DEFINED_LEGAL_TERM",
                "criticality": "CRITICAL",
                "include_in_benchmark": True,
            },
            "occurrences": [{
                "paragraph_id": 0,
                "char_start": 10,
                "char_end": 24,
                "context": "the Effective Date",
            }],
        }],
    }
    translation = {
        "document_id": "doc1",
        "translations": [{"paragraph_id": 0, "translation": "生效日期"}],
    }
    dataset_path = tmp_path / "dataset.json"
    translation_path = tmp_path / "translation.json"
    output_path = tmp_path / "review.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    translation_path.write_text(json.dumps(translation), encoding="utf-8")

    count = locate_term_translations(dataset_path, translation_path, output_path)
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert count == 1
    assert rows[0]["source_term"] == "Effective Date"
    assert rows[0]["criticality"] == "CRITICAL"
    assert rows[0]["target_paragraph"] == "生效日期"


def test_parse_and_validate_rendering_json():
    raw = '''```json
    {"results":[{"occurrence_id":"o1","rendering":"生效日期","status":"FOUND","confidence":0.9,"rationale":"exact"}]}
    ```'''
    parsed = _parse_json(raw)
    batch = [{
        "occurrence_id": "o1",
        "source_term": "Effective Date",
        "target_paragraph": "本协议的生效日期为今日。",
    }]
    result = _validate_results(parsed, batch)
    assert result["results"][0]["rendering"] == "生效日期"

