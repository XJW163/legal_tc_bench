from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_tc_bench import translate as mod


class FakeCompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs["messages"][-1]["content"].split("\n\n", 1)[-1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="译文：" + source[::-1]))]
        )


class FakeOpenAI:
    calls = []

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions(self.calls))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="fake-model")]))


def sample_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "paragraphs": [
                    {"paragraph_id": 0, "text": "Alpha"},
                    {"paragraph_id": 1, "text": "Beta"},
                ],
                "terms": [],
                "benchmark_terms": [],
            }
        ),
        encoding="utf-8",
    )


def test_resumable_single_document(monkeypatch, tmp_path):
    import openai

    FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    src = tmp_path / "doc.json"
    out = tmp_path / "out.json"
    sample_dataset(src)

    first = mod.translate_paragraphs(src, out, "fake-model", "http://fake/v1")
    assert first["status"] == "complete"
    assert len(FakeOpenAI.calls) == 2

    second = mod.translate_paragraphs(src, out, "fake-model", "http://fake/v1")
    assert second["status"] == "complete"
    assert second["statistics"]["restored_paragraphs"] == 2
    assert len(FakeOpenAI.calls) == 2


def test_changed_experiment_rejected(monkeypatch, tmp_path):
    import openai

    FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    src = tmp_path / "doc.json"
    out = tmp_path / "out.json"
    sample_dataset(src)
    mod.translate_paragraphs(src, out, "model-a", "http://fake/v1")
    with pytest.raises(ValueError, match="experiment settings changed"):
        mod.translate_paragraphs(src, out, "model-b", "http://fake/v1")


def test_batch_output_and_validation(monkeypatch, tmp_path):
    import openai

    FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    source_dir = tmp_path / "annotated"
    source_dir.mkdir()
    sample_dataset(source_dir / "doc.json")
    output_root = tmp_path / "translations"

    result = mod.batch_translate(
        source_dir,
        output_root,
        experiment_name="fake-baseline",
        model="fake-model",
        base_url="http://fake/v1",
    )
    assert result["status"] == "complete"
    experiment_dir = output_root / "fake-baseline"
    report = mod.validate_translation_directory(source_dir, experiment_dir)
    assert report["status"] == "valid"
    assert report["missing_paragraphs"] == 0


def test_clean_translation():
    assert mod._clean_translation("```text\n你好\n```") == "你好"
    assert mod._clean_translation("译文：你好") == "你好"
    assert mod._clean_translation("<think>分析 Exhibit</think>\n\n附件10.1") == "附件10.1"
    assert mod._clean_translation("分析 Exhibit\n</think>\n\n附件10.1") == "附件10.1"
    assert mod._clean_translation("第一段</think>第二段</think>最终译文") == "最终译文"


def test_strict_zh_prompt_requires_translation_of_defined_terms(monkeypatch, tmp_path):
    import openai

    FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    src = tmp_path / "doc.json"
    out = tmp_path / "out.json"
    sample_dataset(src)

    result = mod.translate_paragraphs(
        src,
        out,
        "fake-model",
        "http://fake/v1",
        prompt_profile="strict-zh-v1",
    )
    assert result["status"] == "complete"
    request = FakeOpenAI.calls[0]
    assert "defined legal term" in request["messages"][0]["content"]
    assert "不得保留任何未翻译的英文" in request["messages"][-1]["content"]

class ZeroEnglishCompletions:
    def __init__(self, calls):
        self.calls = calls
        self.count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.count += 1
        content = "借款人应向Administrative Agent提交文件"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )]
        )


class ZeroEnglishOpenAI:
    calls = []

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=ZeroEnglishCompletions(self.calls))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="fake-model")]))


def test_zero_english_profile_records_latin_residue_without_retry(monkeypatch, tmp_path):
    import openai

    ZeroEnglishOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", ZeroEnglishOpenAI)
    src = tmp_path / "doc.json"
    out = tmp_path / "out.json"
    sample_dataset(src)

    result = mod.translate_paragraphs(
        src,
        out,
        "fake-model",
        "http://fake/v1",
        prompt_profile="zero-english-v1",
        max_retries=3,
    )
    assert result["status"] == "complete"
    assert len(ZeroEnglishOpenAI.calls) == 2
    for item in result["translations"]:
        assert mod.contains_latin_letters(item["translation"])
        assert item["latin_residue_count"] > 0
        assert item["latin_residue_warning"] is True
        assert item["latin_residue_policy"] == "record"


def test_latin_residue_detection():
    assert mod.contains_latin_letters("行政代理人 Administrative Agent")
    assert not mod.contains_latin_letters("第8.10条第（二）款")
    assert mod.find_latin_residues("借款人 Borrower 与 SEC") == ["Borrower", "SEC"]


def test_validation_treats_latin_residue_as_warning(monkeypatch, tmp_path):
    import openai

    ZeroEnglishOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", ZeroEnglishOpenAI)
    source_dir = tmp_path / "annotated"
    source_dir.mkdir()
    sample_dataset(source_dir / "doc.json")
    output_root = tmp_path / "translations"

    result = mod.batch_translate(
        source_dir,
        output_root,
        experiment_name="fake-zero-english",
        model="fake-model",
        base_url="http://fake/v1",
        prompt_profile="zero-english-v1",
    )
    assert result["status"] == "complete"

    report = mod.validate_translation_directory(
        source_dir,
        output_root / "fake-zero-english",
    )
    assert report["status"] == "valid"
    assert report["quality_status"] == "warning"
    assert report["paragraphs_with_latin_letters"] == 2
    assert report["latin_letter_count"] > 0
