import json
from legal_tc_bench.evaluate import compute_metrics


def test_metrics(tmp_path):
    review = [
        {"term_id": "T1", "source_term": "Purchaser", "rendering": "买方"},
        {"term_id": "T1", "source_term": "Purchaser", "rendering": "买方"},
        {"term_id": "T1", "source_term": "Purchaser", "rendering": "买受人"},
    ]
    inp = tmp_path / "r.json"
    out = tmp_path / "m.json"
    inp.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    compute_metrics(inp, out)
    metrics = json.loads(out.read_text(encoding="utf-8"))["terms"][0]
    assert metrics["dominant_translation_ratio"] == 0.666667
    assert metrics["variant_count"] == 2
