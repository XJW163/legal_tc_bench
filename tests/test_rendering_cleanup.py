import json
from pathlib import Path

from legal_tc_bench.consistency import aggregate_renderings
from legal_tc_bench.rendering_cleanup import (
    seed_reusable_judgments,
    validate_rendering_tree,
)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_validate_flags_overwide_untranslated_and_short(tmp_path):
    paragraph = "Any Person may use Software and such Person shall comply."
    annotated = {
        "document_id": "doc",
        "paragraphs": [{"paragraph_id": 0, "text": paragraph}],
        "benchmark_terms": [
            {
                "term_id": "T1",
                "source_term": "Person",
                "definition_paragraph_ids": [0],
                "occurrences": [
                    {"paragraph_id": 0, "char_start": 4, "char_end": 10},
                    {"paragraph_id": 0, "char_start": 37, "char_end": 43},
                ],
            },
            {
                "term_id": "T2",
                "source_term": "Software",
                "definition_paragraph_ids": [0],
                "occurrences": [{"paragraph_id": 0, "char_start": 19, "char_end": 27}],
            },
        ],
    }
    rows = [
        {"occurrence_id": "1", "document_id": "doc", "term_id": "T1", "source_term": "Person", "paragraph_id": 0, "char_start": 4, "char_end": 10, "target_paragraph": "任何实体可以使用Software。", "rendering": "任何实体", "extraction_status": "FOUND"},
        {"occurrence_id": "2", "document_id": "doc", "term_id": "T2", "source_term": "Software", "paragraph_id": 0, "char_start": 19, "char_end": 27, "target_paragraph": "任何实体可以使用Software。", "rendering": "Software", "extraction_status": "FOUND"},
        {"occurrence_id": "3", "document_id": "doc", "term_id": "T1", "source_term": "Person", "paragraph_id": 0, "char_start": 37, "char_end": 43, "target_paragraph": "该方应遵守。", "rendering": "方", "extraction_status": "FOUND"},
    ]
    ann = tmp_path / "annotated" / "doc.json"
    inp = tmp_path / "renderings" / "doc.review.json"
    out = tmp_path / "validated"
    _write(ann, annotated)
    _write(inp, rows)

    validate_rendering_tree(inp.parent, ann.parent, out)
    result = json.loads((out / inp.name).read_text(encoding="utf-8"))
    assert [r["validation_status"] for r in result] == [
        "POSSIBLY_OVER_WIDE", "UNTRANSLATED", "AMBIGUOUS_ALIGNMENT"
    ]
    assert all(r["needs_rendering_repair"] for r in result)
    assert "<TERM>Person</TERM>" in result[0]["source_marked_context"]


def test_aggregate_excludes_nested_validation_rows(tmp_path):
    rows = [
        {"occurrence_id": "o1", "document_id": "doc", "term_id": "T1", "source_term": "Person", "term_type": "DEFINED_LEGAL_TERM", "criticality": "CRITICAL", "rendering": "主体", "extraction_status": "FOUND", "validation_status": "VALID", "include_in_consistency": True},
        {"occurrence_id": "o2", "document_id": "doc", "term_id": "T1", "source_term": "Person", "term_type": "DEFINED_LEGAL_TERM", "criticality": "CRITICAL", "rendering": "美国人士", "extraction_status": "FOUND", "validation_status": "NESTED_IN_LONGER_TERM", "include_in_consistency": False},
    ]
    inp = tmp_path / "exp" / "doc.review.json"
    out = tmp_path / "agg.json"
    _write(inp, rows)
    aggregate_renderings(inp.parent, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    group = data["groups"][0]
    assert group["raw_total_occurrences"] == 2
    assert group["total_occurrences"] == 1
    assert group["validation_excluded_occurrences"] == 1
    assert group["variant_count"] == 1


def test_reuse_only_unchanged_groups_and_remaps_variant_ids(tmp_path):
    old_agg = {"experiment": "x", "groups": [{"group_id": "g", "source_term": "Term", "total_occurrences": 3, "status_counts": {"FOUND": 3}, "variants": [{"variant_id": "V000", "normalized_rendering": "甲", "count": 2}, {"variant_id": "V001", "normalized_rendering": "乙", "count": 1}]}]}
    old_jud = {"judge_model": "judge", "groups": [{"group_id": "g", "rationale": "group comparison", "variant_judgments": [{"variant_id": "V000", "rendering": "甲", "label": "EXACT_CONSISTENT", "confidence": 1, "rationale": "dominant"}, {"variant_id": "V001", "rendering": "乙", "label": "ACCEPTABLE_VARIANT", "confidence": .9, "rationale": "equivalent"}]}]}
    new_agg = {"experiment": "x", "groups": [{"group_id": "g", "source_term": "Term", "total_occurrences": 3, "status_counts": {"FOUND": 3}, "variants": [{"variant_id": "V000", "normalized_rendering": "甲", "count": 2}, {"variant_id": "V001", "normalized_rendering": "乙", "count": 1}]}]}
    p1, p2, p3, out = [tmp_path / x for x in ["old.agg.json", "old.jud.json", "new.agg.json", "seed.json"]]
    _write(p1, old_agg); _write(p2, old_jud); _write(p3, new_agg)
    summary = seed_reusable_judgments(p1, p2, p3, out)
    assert summary["reused_group_count"] == 1
    seeded = json.loads(out.read_text(encoding="utf-8"))
    assert seeded["groups"][0]["reused_from_previous"] is True


def test_validate_excludes_occurrence_nested_in_longer_term(tmp_path):
    paragraph = "A U.S. Person shall comply."
    annotated = {
        "document_id": "doc",
        "paragraphs": [{"paragraph_id": 0, "text": paragraph}],
        "benchmark_terms": [
            {"term_id": "T1", "source_term": "Person", "definition_paragraph_ids": [], "occurrences": [{"paragraph_id": 0, "char_start": 7, "char_end": 13}]},
            {"term_id": "T2", "source_term": "U.S. Person", "definition_paragraph_ids": [], "occurrences": [{"paragraph_id": 0, "char_start": 2, "char_end": 13}]},
        ],
    }
    rows = [{"occurrence_id": "o", "document_id": "doc", "term_id": "T1", "source_term": "Person", "paragraph_id": 0, "char_start": 7, "char_end": 13, "target_paragraph": "美国人士应遵守。", "rendering": "美国人士", "extraction_status": "FOUND"}]
    ann = tmp_path / "ann" / "doc.json"; inp = tmp_path / "in" / "doc.review.json"; out = tmp_path / "out"
    _write(ann, annotated); _write(inp, rows)
    validate_rendering_tree(inp.parent, ann.parent, out)
    result = json.loads((out / inp.name).read_text(encoding="utf-8"))[0]
    assert result["validation_status"] == "NESTED_IN_LONGER_TERM"
    assert result["include_in_consistency"] is False
    assert result["containing_term_id"] == "T2"


def test_repair_batch_robust_retains_failed_single(monkeypatch):
    from legal_tc_bench import rendering_cleanup as module

    row = {
        "occurrence_id": "o1",
        "rendering": "任何实体",
        "target_paragraph": "任何实体应遵守。",
    }

    def fail(*args, **kwargs):
        raise ValueError("Model returned empty text")

    monkeypatch.setattr(module, "_call_repair_with_retry", fail)
    results, raw = module._repair_batch_robust(object(), "judge", [row], 5, 1200)
    assert results[0]["status"] == "RETAIN_ORIGINAL_REVIEW"
    assert results[0]["rendering"] == "任何实体"
    assert "REPAIR_FAILED_RETAINED" in raw


def test_repair_batch_robust_isolates_bad_row(monkeypatch):
    from legal_tc_bench import rendering_cleanup as module

    rows = [
        {"occurrence_id": "good", "rendering": "实体", "target_paragraph": "实体应遵守。"},
        {"occurrence_id": "bad", "rendering": "任何主体", "target_paragraph": "任何主体应遵守。"},
    ]

    def fake(client, model, batch, max_retries, max_output_tokens):
        oid = batch[0]["occurrence_id"] if len(batch) == 1 else "batch"
        if len(batch) > 1 or oid == "bad":
            raise ValueError("invalid model output")
        return ([{
            "occurrence_id": "good",
            "rendering": "实体",
            "status": "FOUND_MINIMAL",
            "confidence": 0.9,
            "rationale": "exact head",
        }], "{}")

    monkeypatch.setattr(module, "_call_repair_with_retry", fake)
    results, _ = module._repair_batch_robust(object(), "judge", rows, 5, 1200)
    by_id = {r["occurrence_id"]: r for r in results}
    assert by_id["good"]["status"] == "FOUND_MINIMAL"
    assert by_id["bad"]["status"] == "RETAIN_ORIGINAL_REVIEW"


def test_apply_failed_repair_preserves_and_excludes():
    from legal_tc_bench.rendering_cleanup import _apply_repair_results

    row = {
        "occurrence_id": "o1",
        "rendering": "任何实体",
        "validation_status": "POSSIBLY_OVER_WIDE",
        "needs_rendering_repair": True,
        "include_in_consistency": True,
    }
    result = [{
        "occurrence_id": "o1",
        "rendering": "任何实体",
        "status": "RETAIN_ORIGINAL_REVIEW",
        "confidence": 0.0,
        "rationale": "model output unusable",
    }]
    _apply_repair_results([row], result, "judge")
    assert row["rendering"] == "任何实体"
    assert row["validation_status"] == "REPAIR_FAILED_RETAINED"
    assert row["include_in_consistency"] is False
    assert row["needs_rendering_repair"] is True


def test_recover_exact_substring_ignores_whitespace_only():
    from legal_tc_bench.rendering_cleanup import _recover_exact_substring

    assert _recover_exact_substring("贷款文件", "相关贷款 文件应当交付") == "贷款 文件"
    assert _recover_exact_substring("贷款文件", "相关贷款文档应当交付") is None
