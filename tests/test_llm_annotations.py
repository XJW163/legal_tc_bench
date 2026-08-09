from legal_tc_bench.llm_annotations import _parse_json, _validate_results


def test_validate_llm_results():
    value = {
        "results": [{
            "item_id": 0,
            "term_type": "DEFINED_LEGAL_TERM",
            "criticality": "CRITICAL",
            "include_in_benchmark": True,
            "confidence": 0.95,
            "rationale": "Explicitly defined effective date.",
        }]
    }
    _validate_results(value, 1)


def test_parse_fenced_json():
    value = _parse_json('```json\n{"results": []}\n```')
    assert value == {"results": []}
