from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from legal_tc_bench import term_memory as mod


class FakeCompletions:
    def create(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        results = []
        for item in payload["items"]:
            results.append({
                "item_id": item["item_id"],
                "canonical_translation": "主体" if item["source_term"] == "Person" else "生效日期",
                "legal_scope_summary": "依据合同定义覆盖相关法律主体或时间概念。",
                "allowed_variants": [{"rendering": "实体", "condition": "不缩小定义范围时"}] if item["source_term"] == "Person" else [],
                "risky_variants": [{"rendering": "个人", "risk_type": "SCOPE_NARROWING", "reason": "可能缩小为自然人"}] if item["source_term"] == "Person" else [],
                "confidence": 0.95,
                "rationale": "根据定义条款选择稳定译法。",
            })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"results": results}, ensure_ascii=False)))]
        )


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _annotated(path: Path):
    path.write_text(json.dumps({
        "document_id": "doc-1",
        "paragraphs": [
            {"paragraph_id": 0, "text": '"Person" means any individual, corporation or partnership.'},
            {"paragraph_id": 1, "text": "Any Person may sign on the Effective Date."},
        ],
        "benchmark_terms": [
            {
                "term_id": "T0001",
                "source_term": "Person",
                "definition_paragraph_ids": [0],
                "occurrences": [{"paragraph_id": 1, "char_start": 4, "char_end": 10, "context": "Any Person may sign"}],
                "annotation": {"term_type": "DEFINED_LEGAL_TERM", "criticality": "CRITICAL"},
            },
            {
                "term_id": "T0002",
                "source_term": "Effective Date",
                "definition_paragraph_ids": [],
                "occurrences": [{"paragraph_id": 1, "char_start": 27, "char_end": 41, "context": "on the Effective Date"}],
                "annotation": {"term_type": "DEFINED_LEGAL_TERM", "criticality": "IMPORTANT"},
            },
        ],
    }, ensure_ascii=False), encoding="utf-8")


def test_build_term_memory(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace as SNS
    fake_module = SNS(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    annotated = tmp_path / "doc.json"
    _annotated(annotated)
    output = tmp_path / "memory"
    result = mod.build_term_memory_tree(
        annotated,
        output,
        model="fake",
        base_url="http://fake/v1",
        batch_size=2,
    )
    assert result["status"] == "complete"
    data = json.loads((output / "doc-1.term-memory.json").read_text(encoding="utf-8"))
    assert data["completed_term_count"] == 2
    person = {x["term_id"]: x for x in data["terms"]}["T0001"]
    assert person["canonical_translation"] == "主体"
    assert person["risky_variants"][0]["risk_type"] == "SCOPE_NARROWING"
