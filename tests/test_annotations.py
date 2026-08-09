import csv
import json
from pathlib import Path

from legal_tc_bench.annotations import apply_term_review, export_term_review, suggest_annotation


def sample_doc():
    return {
        "schema_version": "1.0",
        "document_id": "doc1",
        "paragraphs": [{"paragraph_id": 0, "text": 'This Agreement ("Agreement") starts on the Effective Date.'}],
        "terms": [
            {"term_id": "T0001", "source_term": "Agreement", "extraction_method": "x", "definition_paragraph_ids": [0], "occurrences": [{}, {}]},
            {"term_id": "T0002", "source_term": "Effective Date", "extraction_method": "x", "definition_paragraph_ids": [0], "occurrences": [{}]},
        ],
    }


def test_export_and_apply(tmp_path: Path):
    src = tmp_path / "doc.json"
    src.write_text(json.dumps(sample_doc()), encoding="utf-8")
    review = tmp_path / "review.csv"
    assert export_term_review(src, review) == 2

    rows = list(csv.DictReader(review.open(encoding="utf-8-sig")))
    for row in rows:
        row["term_type"] = row["suggested_term_type"]
        row["criticality"] = row["suggested_criticality"]
        row["include_in_benchmark"] = row["suggested_include"]
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    out = tmp_path / "annotated.json"
    result = apply_term_review(src, review, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    assert len(data["benchmark_terms"]) == 2
    assert result["totals"]["reviewed"] == 2


def test_suggestion_party_acronym():
    assert suggest_annotation("DS") == ("PARTY_NAME", "IMPORTANT", False)
