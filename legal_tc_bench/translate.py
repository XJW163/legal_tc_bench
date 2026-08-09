from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from .io_utils import atomic_write_json

import httpx
from tqdm import tqdm


DEFAULT_BASE_URL = "http://10.12.143.51:8000/v1"
DEFAULT_MODEL = "gpt-oss-120b"

PROMPT_PROFILES: dict[str, str] = {
    "baseline-v1": (
        "You are a professional English-to-Simplified-Chinese legal translator. "
        "Translate the supplied legal-contract paragraph accurately and completely. "
        "Preserve section numbering, dates, amounts, party names, placeholders such as [***], "
        "defined-term capitalization distinctions, and cross-references. "
        "Do not summarize, omit, merge, explain, or add content. "
        "Return only the Chinese translation of the current paragraph."
    ),
    "consistency-aware-v1": (
        "You are a professional English-to-Simplified-Chinese legal translator. "
        "Translate the supplied legal-contract paragraph accurately and completely. "
        "Use stable Chinese renderings for the same defined legal terms throughout the document. "
        "Preserve section numbering, dates, amounts, party names, placeholders such as [***], "
        "defined-term capitalization distinctions, and cross-references. "
        "Do not summarize, omit, merge, explain, or add content. "
        "Return only the Chinese translation of the current paragraph."
    ),
    "strict-zh-v1": (
        "You are a professional English-to-Simplified-Chinese legal translator. "
        "Translate the entire supplied legal-contract paragraph into natural, professional Simplified Chinese. "
        "Mandatory requirements: translate every ordinary English word, phrase, sentence, contractual role, "
        "defined legal term, legal concept, right, obligation, condition, event, and document name into Chinese. "
        "Do not retain an English term merely because it is capitalized, quoted, or defined by the contract. "
        "Defined terms must be rendered as Chinese defined terms, not left in English and not shown bilingually. "
        "Only genuine personal names, registered company names, product names, standard abbreviations, URLs, "
        "email addresses, statute citations, clause or section numbers, and placeholders such as [***] may remain "
        "in their original form when appropriate. Preserve all numbering, dates, amounts, subsection markers, "
        "cross-references, conditions, exceptions, and legal effects. Do not summarize, omit, merge, explain, "
        "comment on, or add content. Do not reproduce the English source. Return only the complete Chinese translation."
    ),
    "zero-english-v1": (
        "You are a professional English-to-Simplified-Chinese legal translator. "
        "Translate the entire supplied legal-contract paragraph into professional Simplified Chinese with ZERO Latin letters in the output. "
        "Every English word, phrase, sentence, defined term, contractual role, legal concept, document title, company name, personal name, "
        "product name, abbreviation, acronym, alphabetic subsection marker, and English cross-reference component must be translated, "
        "transliterated, or rewritten in Chinese. Do not leave any A-Z or a-z character anywhere in the answer. "
        "For example, translate Administrative Agent as 行政代理人, Borrower as 借款人, Material Adverse Effect as 重大不利影响, "
        "and rewrite Section 8.10.(b) as 第8.10条第（二）款. Translate company and personal names by established Chinese names or phonetic transliteration. "
        "Translate abbreviations by their full Chinese meaning rather than retaining the letters. Preserve Arabic numerals, dates, monetary values, "
        "mathematical symbols, punctuation, and placeholders such as [***]. Do not output bilingual text, explanations, notes, or the English source. "
        "Do not summarize, omit, merge, or add substantive content. Return only the complete Chinese translation, containing no Latin letters."
    ),
}

USER_PROMPT_PROFILES: dict[str, str] = {
    "baseline-v1": "Translate this paragraph into Simplified Chinese. Return only the translation:\n\n{source_text}",
    "consistency-aware-v1": "Translate this paragraph into Simplified Chinese. Return only the translation:\n\n{source_text}",
    "strict-zh-v1": (
        "请将以下英文法律合同段落完整翻译为专业、自然的简体中文。\n"
        "除真实的人名、注册公司名称、产品名、标准缩写、网址、邮箱、法规编号、条款号及[***]占位符外，"
        "不得保留任何未翻译的英文单词、短语、定义术语或句子。\n"
        "合同角色、定义术语、法律概念、文件名称、条件和事件均必须译成中文；不得输出中英对照，"
        "不得在中文术语后附加英文原词。只输出完整中文译文。\n\n英文原文：\n{source_text}"
    ),
    "zero-english-v1": (
        "请将以下英文法律合同段落全部翻译为专业简体中文，输出中不得出现任何英文字母。\n"
        "具体要求：\n"
        "一、所有合同角色、定义术语、法律概念、文件名称、公司名称、人名、产品名称、缩写和首字母简称均须翻译或音译成中文；\n"
        "二、不得保留任何大写或小写拉丁字母，不得输出中英对照，也不得在中文译名后附英文；\n"
        "三、英文字母条款标记和交叉引用也要改写为中文层级，例如 Section 8.10.(b) 写成“第8.10条第（二）款”；\n"
        "四、阿拉伯数字、日期、金额、数学符号、标点和[***]占位符可以保留；\n"
        "五、不得概括、删减、合并、解释或增加实质内容。\n"
        "只输出完整中文译文，并确保最终结果中不存在任何A至Z或a至z字母。\n\n英文原文：\n{source_text}"
    ),
}

PROFILE_TASK_REQUIREMENTS: dict[str, str] = {
    "baseline-v1": "请准确、完整地翻译当前段落。",
    "consistency-aware-v1": "请准确、完整地翻译当前段落，并与前文既有译文保持术语一致。",
    "strict-zh-v1": (
        "除真实人名、注册公司名称、产品名、标准缩写、网址、邮箱、法规编号、条款号及[***]占位符外，"
        "不得保留任何未翻译的英文单词、短语、定义术语或句子；不得输出中英对照。"
    ),
    "zero-english-v1": (
        "输出中不得出现任何大写或小写拉丁字母。所有公司名称、人名、缩写、定义术语和英文字母条款标记"
        "均须翻译、音译或改写为中文；不得输出中英对照。"
    ),
}


LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[A-Za-z0-9._:/@+\-]*[A-Za-z0-9])?")
ZERO_ENGLISH_PROFILES = {"zero-english-v1"}


def find_latin_residues(text: str, limit: int = 30) -> list[str]:
    """Return distinct Latin-containing tokens in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for match in LATIN_TOKEN_RE.finditer(text):
        token = match.group(0)
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def contains_latin_letters(text: str) -> bool:
    return bool(LATIN_LETTER_RE.search(text))


@dataclass(frozen=True)
class TranslationExperiment:
    name: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    prompt_profile: str = "baseline-v1"
    temperature: float = 0.0
    max_tokens: int = 4096
    source_chunk_chars: int = 12000
    context_paragraphs: int = 2
    max_context_chars: int = 12000
    extra_body: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        if self.prompt_profile not in PROMPT_PROFILES:
            raise ValueError(
                f"unknown prompt profile {self.prompt_profile!r}; "
                f"choose from {sorted(PROMPT_PROFILES)}"
            )
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.source_chunk_chars < 1000:
            raise ValueError("source_chunk_chars must be >= 1000")
        if self.context_paragraphs < 0:
            raise ValueError("context_paragraphs must be >= 0")
        if self.max_context_chars < 0:
            raise ValueError("max_context_chars must be >= 0")


class TranslationError(RuntimeError):
    pass


class TranslationTruncatedError(TranslationError):
    """Raised only when the model stops because max_tokens was exhausted."""



def check_openai_compatible_server(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 30.0,
) -> list[str]:
    """Return model IDs advertised by an OpenAI-compatible server."""
    from openai import OpenAI

    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(
            connect=min(30.0, timeout_seconds),
            read=timeout_seconds,
            write=timeout_seconds,
            pool=min(30.0, timeout_seconds),
        ),
    )
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )
        return [str(x.id) for x in client.models.list().data]
    finally:
        http_client.close()


def translate_paragraphs(
    dataset_path: Path,
    output_path: Path,
    model: str,
    base_url: str | None = None,
    *,
    prompt_profile: str = "baseline-v1",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    source_chunk_chars: int = 12000,
    context_paragraphs: int = 2,
    max_context_chars: int = 12000,
    extra_body: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout_seconds: float = 300.0,
    overwrite: bool = False,
    retry_errors_only: bool = False,
) -> dict[str, Any]:
    """Translate one structured dataset JSON with paragraph-level resume."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    dataset = _read_dataset(dataset_path)
    experiment = TranslationExperiment(
        name=output_path.stem,
        model=model,
        base_url=base_url or DEFAULT_BASE_URL,
        prompt_profile=prompt_profile,
        temperature=temperature,
        max_tokens=max_tokens,
        source_chunk_chars=source_chunk_chars,
        context_paragraphs=context_paragraphs,
        max_context_chars=max_context_chars,
        extra_body=extra_body,
    )
    result = _translate_document(
        dataset=dataset,
        source_path=dataset_path,
        output_path=output_path,
        experiment=experiment,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        retry_errors_only=retry_errors_only,
    )
    return result


def batch_translate(
    input_path: Path,
    output_root: Path,
    *,
    experiment_name: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    prompt_profile: str = "baseline-v1",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    source_chunk_chars: int = 12000,
    context_paragraphs: int = 2,
    max_context_chars: int = 12000,
    extra_body: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout_seconds: float = 300.0,
    overwrite: bool = False,
    continue_on_error: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Translate all dataset documents into an isolated experiment directory.

    The output itself is the checkpoint. Every successful paragraph is written
    atomically. Re-running the same command skips valid completed paragraphs.
    """
    experiment = TranslationExperiment(
        name=experiment_name,
        model=model,
        base_url=base_url,
        prompt_profile=prompt_profile,
        temperature=temperature,
        max_tokens=max_tokens,
        source_chunk_chars=source_chunk_chars,
        context_paragraphs=context_paragraphs,
        max_context_chars=max_context_chars,
        extra_body=extra_body,
    )
    experiment.validate()
    if workers < 1:
        raise ValueError("workers must be >= 1")

    documents = list(_iter_dataset_documents(Path(input_path)))
    experiment_dir = Path(output_root) / _safe_slug(experiment.name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_dir / "translation_manifest.json"

    manifest = _load_json_or_default(manifest_path, {})
    manifest.update({
        "schema_version": "1.0",
        "experiment": _experiment_metadata(experiment),
        "input": str(input_path),
        "output_directory": str(experiment_dir),
        "total_documents": len(documents),
        "documents": manifest.get("documents", {}),
        "started_at": manifest.get("started_at") or _now(),
        "updated_at": _now(),
        "status": "in_progress",
        "workers": workers,
    })
    _atomic_json(manifest_path, manifest)

    totals = {
        "documents_total": len(documents),
        "documents_complete": 0,
        "documents_incomplete": 0,
        "paragraphs_total": 0,
        "paragraphs_translated": 0,
        "paragraphs_restored": 0,
        "paragraph_errors": 0,
        "errors": 0,
    }

    def process_document(source_path: Path, dataset: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
        document_id = str(dataset.get("document_id") or source_path.stem)
        destination = experiment_dir / f"{_safe_slug(document_id)}.translation.json"
        result = _translate_document(
            dataset=dataset,
            source_path=source_path,
            output_path=destination,
            experiment=experiment,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
            show_progress=(workers == 1),
        )
        return document_id, destination, result

    def record_success(
        source_path: Path,
        document_id: str,
        destination: Path,
        result: dict[str, Any],
    ) -> None:
        totals["paragraphs_total"] += int(result["statistics"]["total_paragraphs"])
        totals["paragraphs_translated"] += int(result["statistics"]["translated_paragraphs"])
        totals["paragraphs_restored"] += int(result["statistics"].get("restored_paragraphs", 0))
        totals["paragraph_errors"] += int(result["statistics"].get("failed_paragraphs", 0))
        if result["status"] == "complete":
            totals["documents_complete"] += 1
        else:
            totals["documents_incomplete"] += 1
        manifest["documents"][document_id] = {
            "source": str(source_path),
            "output": str(destination),
            "status": result["status"],
            "total_paragraphs": result["statistics"]["total_paragraphs"],
            "translated_paragraphs": result["statistics"]["translated_paragraphs"],
            "failed_paragraphs": result["statistics"].get("failed_paragraphs", 0),
            "updated_at": result["updated_at"],
        }

    def record_failure(source_path: Path, dataset: dict[str, Any], exc: Exception) -> None:
        document_id = str(dataset.get("document_id") or source_path.stem)
        destination = experiment_dir / f"{_safe_slug(document_id)}.translation.json"
        totals["errors"] += 1
        totals["documents_incomplete"] += 1
        manifest["documents"][document_id] = {
            "source": str(source_path),
            "output": str(destination),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at": _now(),
        }
        tqdm.write(f"translation failed for {document_id}: {exc}")

    progress = tqdm(total=len(documents), desc=f"translate {experiment.name}", unit="doc", dynamic_ncols=True)
    try:
        if workers == 1:
            for source_path, dataset in documents:
                try:
                    document_id, destination, result = process_document(source_path, dataset)
                    record_success(source_path, document_id, destination, result)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    record_failure(source_path, dataset, exc)
                    if not continue_on_error:
                        manifest["status"] = "stopped_on_error"
                        manifest["updated_at"] = _now()
                        manifest["totals"] = totals
                        _atomic_json(manifest_path, manifest)
                        raise
                finally:
                    progress.update(1)
                    manifest["updated_at"] = _now()
                    manifest["totals"] = totals
                    _atomic_json(manifest_path, manifest)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="legal-tc") as executor:
                future_map: dict[Future[tuple[str, Path, dict[str, Any]]], tuple[Path, dict[str, Any]]] = {
                    executor.submit(process_document, source_path, dataset): (source_path, dataset)
                    for source_path, dataset in documents
                }
                first_error: Exception | None = None
                for future in as_completed(future_map):
                    source_path, dataset = future_map[future]
                    try:
                        document_id, destination, result = future.result()
                        record_success(source_path, document_id, destination, result)
                    except CancelledError:
                        continue
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        record_failure(source_path, dataset, exc)
                        if first_error is None:
                            first_error = exc
                        if not continue_on_error:
                            for pending in future_map:
                                pending.cancel()
                    finally:
                        progress.update(1)
                        manifest["updated_at"] = _now()
                        manifest["totals"] = totals
                        _atomic_json(manifest_path, manifest)
                if first_error is not None and not continue_on_error:
                    manifest["status"] = "stopped_on_error"
                    manifest["updated_at"] = _now()
                    manifest["totals"] = totals
                    _atomic_json(manifest_path, manifest)
                    raise first_error
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _now()
        manifest["totals"] = totals
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        progress.close()

    manifest["status"] = "complete" if totals["documents_complete"] == len(documents) else "partial"
    manifest["updated_at"] = _now()
    manifest["totals"] = totals
    _atomic_json(manifest_path, manifest)
    return {"experiment_directory": str(experiment_dir), "manifest": str(manifest_path), **totals, "status": manifest["status"]}


def run_translation_matrix(
    config_path: Path,
    *,
    input_override: Path | None = None,
    output_override: Path | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run multiple model/prompt experiments from a reproducible JSON config."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = input_override or Path(config.get("input", "data/annotated"))
    output_root = output_override or Path(config.get("output", "data/translations"))
    defaults = config.get("defaults", {})
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("translation matrix config must contain a non-empty experiments list")

    results: list[dict[str, Any]] = []
    for raw in experiments:
        if raw.get("enabled", True) is False:
            continue
        merged = {**defaults, **raw}
        name = str(merged.get("name", "")).strip()
        if not name:
            raise ValueError("every enabled experiment requires a name")
        try:
            result = batch_translate(
                input_path,
                output_root,
                experiment_name=name,
                model=str(merged.get("model", DEFAULT_MODEL)),
                base_url=str(merged.get("base_url", DEFAULT_BASE_URL)),
                prompt_profile=str(merged.get("prompt_profile", "baseline-v1")),
                temperature=float(merged.get("temperature", 0.0)),
                max_tokens=int(merged.get("max_tokens", 4096)),
                source_chunk_chars=int(merged.get("source_chunk_chars", 12000)),
                context_paragraphs=int(merged.get("context_paragraphs", 0)),
                max_context_chars=int(merged.get("max_context_chars", 12000)),
                extra_body=merged.get("extra_body") if isinstance(merged.get("extra_body"), dict) else None,
                max_retries=int(merged.get("max_retries", 5)),
                timeout_seconds=float(merged.get("timeout_seconds", 300.0)),
                overwrite=overwrite,
                continue_on_error=continue_on_error,
                workers=int(merged.get("workers", 1)),
            )
            results.append({"name": name, "status": result["status"], "result": result})
        except Exception as exc:
            results.append({"name": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            if not continue_on_error:
                break

    report = {
        "schema_version": "1.0",
        "config": str(config_path),
        "input": str(input_path),
        "output": str(output_root),
        "created_at": _now(),
        "results": results,
    }
    report_path = Path(output_root) / "translation_matrix_report.json"
    _atomic_json(report_path, report)
    return {"report": str(report_path), "experiments": results}


def validate_translation_directory(
    input_path: Path,
    experiment_directory: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate document/paragraph coverage without calling any model."""
    source_docs = {str(doc.get("document_id") or path.stem): (path, doc) for path, doc in _iter_dataset_documents(Path(input_path))}
    translations: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(Path(experiment_directory).glob("*.translation.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        document_id = str(data.get("document_id") or "")
        if document_id:
            translations[document_id] = (path, data)

    report: dict[str, Any] = {
        "input": str(input_path),
        "experiment_directory": str(experiment_directory),
        "source_documents": len(source_docs),
        "translation_documents": len(translations),
        "missing_documents": [],
        "extra_documents": [],
        "documents": {},
        "totals": {
            "source_paragraphs": 0,
            "translated_paragraphs": 0,
            "missing_paragraphs": 0,
            "empty_translations": 0,
            "source_mismatches": 0,
            "paragraphs_with_latin_letters": 0,
            "latin_letter_count": 0,
        },
    }

    for document_id, (_, source) in source_docs.items():
        paragraphs = source.get("paragraphs", [])
        report["totals"]["source_paragraphs"] += len(paragraphs)
        if document_id not in translations:
            report["missing_documents"].append(document_id)
            report["totals"]["missing_paragraphs"] += len(paragraphs)
            continue
        path, target = translations[document_id]
        requires_zero_english = (
            str(target.get("experiment", {}).get("prompt_profile", "")) in ZERO_ENGLISH_PROFILES
        )
        by_id = {str(x.get("paragraph_id")): x for x in target.get("translations", []) if isinstance(x, dict)}
        missing: list[Any] = []
        empty: list[Any] = []
        mismatched: list[Any] = []
        latin_paragraphs: list[Any] = []
        latin_count = 0
        for paragraph in paragraphs:
            pid = str(paragraph.get("paragraph_id"))
            item = by_id.get(pid)
            if item is None:
                missing.append(paragraph.get("paragraph_id"))
                continue
            report["totals"]["translated_paragraphs"] += 1
            if not str(item.get("translation", "")).strip():
                empty.append(paragraph.get("paragraph_id"))
            target_text = str(item.get("translation", ""))
            if requires_zero_english and contains_latin_letters(target_text):
                latin_paragraphs.append(paragraph.get("paragraph_id"))
                latin_count += len(LATIN_LETTER_RE.findall(target_text))
            if str(item.get("source", "")) != str(paragraph.get("text", "")):
                mismatched.append(paragraph.get("paragraph_id"))
        report["totals"]["missing_paragraphs"] += len(missing)
        report["totals"]["empty_translations"] += len(empty)
        report["totals"]["source_mismatches"] += len(mismatched)
        report["totals"]["paragraphs_with_latin_letters"] += len(latin_paragraphs)
        report["totals"]["latin_letter_count"] += latin_count
        report["documents"][document_id] = {
            "translation_file": str(path),
            "source_paragraphs": len(paragraphs),
            "translated_paragraphs": len(by_id),
            "missing_paragraph_ids": missing,
            "empty_paragraph_ids": empty,
            "source_mismatch_paragraph_ids": mismatched,
            "latin_residue_paragraph_ids": latin_paragraphs,
            "latin_letter_count": latin_count,
            "zero_english_required": requires_zero_english,
        }

    report["extra_documents"] = sorted(set(translations) - set(source_docs))
    structural_valid = (
        not report["missing_documents"]
        and not report["extra_documents"]
        and report["totals"]["missing_paragraphs"] == 0
        and report["totals"]["empty_translations"] == 0
        and report["totals"]["source_mismatches"] == 0
    )
    report["status"] = "valid" if structural_valid else "invalid"
    report["quality_status"] = (
        "warning"
        if report["totals"]["paragraphs_with_latin_letters"] > 0
        else "clean"
    )
    report["warnings"] = []
    if report["totals"]["paragraphs_with_latin_letters"] > 0:
        report["warnings"].append({
            "type": "latin_residue",
            "paragraphs": report["totals"]["paragraphs_with_latin_letters"],
            "letters": report["totals"]["latin_letter_count"],
            "message": (
                "Latin letters were detected and recorded as a quality warning. "
                "They do not make the translation structurally invalid."
            ),
        })
    report["created_at"] = _now()

    destination = output_path or Path(experiment_directory) / "validation_report.json"
    _atomic_json(destination, report)
    return {
        "report": str(destination),
        "status": report["status"],
        "quality_status": report["quality_status"],
        **report["totals"],
        "missing_documents": len(report["missing_documents"]),
        "extra_documents": len(report["extra_documents"]),
    }


def _translate_document(
    *,
    dataset: dict[str, Any],
    source_path: Path,
    output_path: Path,
    experiment: TranslationExperiment,
    max_retries: int,
    timeout_seconds: float,
    overwrite: bool,
    retry_errors_only: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    experiment.validate()
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    document_id = str(dataset.get("document_id") or source_path.stem)
    paragraphs = dataset.get("paragraphs")
    if not isinstance(paragraphs, list):
        raise ValueError(f"{source_path}: missing paragraphs list")
    source_signature = _source_signature(dataset)
    experiment_meta = _experiment_metadata(experiment)
    experiment_signature = _json_hash(experiment_meta)

    existing: dict[str, Any] = {}
    if output_path.exists() and not overwrite:
        existing = _load_json_or_default(output_path, {})
        if existing:
            if str(existing.get("document_id")) != document_id:
                raise ValueError(f"{output_path}: document_id does not match {document_id}")
            if existing.get("source_signature") != source_signature:
                raise ValueError(f"{output_path}: source document changed; use --overwrite")
            if existing.get("experiment_signature") != experiment_signature:
                if not retry_errors_only:
                    raise ValueError(
                        f"{output_path}: experiment settings changed; "
                        "use another experiment name or --overwrite"
                    )
                # Error-only retry is a repair operation on an existing result.
                # Permit it even when the current code's prompt metadata/hash has
                # changed, because all successful paragraphs remain untouched and
                # only paragraph IDs already present in `errors` are sent again.

    if retry_errors_only and overwrite:
        raise ValueError("--retry-errors-only cannot be used with --overwrite")
    if retry_errors_only and not existing:
        raise ValueError("--retry-errors-only requires an existing translation result file")

    retry_error_ids = {
        str(item.get("paragraph_id"))
        for item in existing.get("errors", [])
        if isinstance(item, dict) and item.get("paragraph_id") is not None
    }
    if retry_errors_only and not retry_error_ids:
        # Nothing failed previously. Return the existing result unchanged.
        return existing

    restored: dict[str, dict[str, Any]] = {}
    for item in existing.get("translations", []) if isinstance(existing, dict) else []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paragraph_id"))
        source = str(item.get("source", ""))
        translation = str(item.get("translation", ""))
        if translation.strip() and item.get("source_hash") == _text_hash(source):
            restored[pid] = item

    # Error-only retry is an in-place repair operation. Start from the full
    # existing translation set so every checkpoint preserves paragraphs that
    # occur after the first failed paragraph.
    existing_translations = [
        item for item in existing.get("translations", [])
        if isinstance(item, dict)
    ] if existing else []

    state: dict[str, Any] = {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_path": str(source_path),
        "source_signature": source_signature,
        "experiment": experiment_meta,
        "experiment_signature": experiment_signature,
        "translations": list(existing_translations) if retry_errors_only else [],
        "errors": existing.get("errors", []) if existing else [],
        "status": "in_progress",
        "started_at": existing.get("started_at") if existing else _now(),
        "updated_at": _now(),
        "statistics": {},
    }

    from openai import OpenAI

    timeout = httpx.Timeout(
        connect=min(30.0, timeout_seconds),
        read=timeout_seconds,
        write=min(120.0, timeout_seconds),
        pool=min(30.0, timeout_seconds),
    )
    http_client = httpx.Client(trust_env=False, timeout=timeout)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=experiment.base_url,
        http_client=http_client,
        max_retries=0,
    )

    restored_count = 0
    translated_count = 0
    ordered_results: list[dict[str, Any]] = []

    # In repair mode, keep a complete paragraph-id keyed copy of all existing
    # successful translations. Updating one failed paragraph replaces only that
    # entry; it never rebuilds the output from a partial prefix.
    repair_results: dict[str, dict[str, Any]] = {
        str(item.get("paragraph_id")): item
        for item in existing_translations
        if item.get("paragraph_id") is not None
    }

    def current_repair_results() -> list[dict[str, Any]]:
        if not retry_errors_only:
            return ordered_results
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            paragraph_id = str(paragraph.get("paragraph_id"))
            item = repair_results.get(paragraph_id)
            if item is not None:
                ordered.append(item)
                seen.add(paragraph_id)
        # Preserve any legacy/extra records not present in the current dataset.
        ordered.extend(item for pid, item in repair_results.items() if pid not in seen)
        return ordered
    try:
        progress = tqdm(
            paragraphs,
            desc=f"{document_id}",
            unit="para",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for index, paragraph in enumerate(progress):
            pid = str(paragraph.get("paragraph_id"))
            source_text = str(paragraph.get("text", ""))
            saved = restored.get(pid)

            # In error-only retry mode, paragraphs that were not recorded as
            # failures must never be sent to the model again. Existing successful
            # translations are restored verbatim and missing non-error items are
            # left untouched.
            if retry_errors_only and pid not in retry_error_ids:
                continue

            # A paragraph listed in errors must be retried even if a stale
            # translation record with the same paragraph_id also exists.
            if not retry_errors_only and saved is not None and saved.get("source") == source_text:
                ordered_results.append(saved)
                restored_count += 1
                state["translations"] = ordered_results
                state["errors"] = [
                    x for x in state["errors"]
                    if str(x.get("paragraph_id")) != pid
                ]
                continue

            if not source_text.strip():
                item = {
                    "paragraph_id": paragraph.get("paragraph_id"),
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "translation": "",
                    "model": experiment.model,
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "completed_at": _now(),
                }
            else:
                context = _build_previous_context(
                    paragraphs=paragraphs,
                    completed=ordered_results,
                    current_index=index,
                    count=experiment.context_paragraphs,
                    max_chars=experiment.max_context_chars,
                )
                try:
                    translation, attempts, latency, chunk_count = _translate_source_text(
                        client=client,
                        model=experiment.model,
                        prompt_profile=experiment.prompt_profile,
                        source_text=source_text,
                        previous_context=context,
                        temperature=experiment.temperature,
                        max_tokens=experiment.max_tokens,
                        max_retries=max_retries,
                        source_chunk_chars=experiment.source_chunk_chars,
                        extra_body=experiment.extra_body,
                    )
                except Exception as exc:
                    error = {
                        "paragraph_id": paragraph.get("paragraph_id"),
                        "source": source_text,
                        "source_hash": _text_hash(source_text),
                        "error": f"{type(exc).__name__}: {exc}",
                        "time": _now(),
                    }
                    state["errors"] = [
                        x for x in state["errors"]
                        if str(x.get("paragraph_id")) != pid
                    ]
                    state["errors"].append(error)
                    checkpoint_results = current_repair_results()
                    state["translations"] = checkpoint_results
                    state["statistics"] = _translation_statistics(
                        paragraphs,
                        checkpoint_results,
                        restored_count,
                        state["errors"],
                    )
                    state["status"] = "incomplete"
                    state["updated_at"] = _now()
                    _atomic_json(output_path, state)
                    tqdm.write(
                        f"paragraph failed; continuing: "
                        f"{document_id} paragraph {pid}: {exc}"
                    )
                    continue

                item = {
                    "paragraph_id": paragraph.get("paragraph_id"),
                    "source": source_text,
                    "source_hash": _text_hash(source_text),
                    "translation": translation,
                    "model": experiment.model,
                    "attempts": attempts,
                    "latency_seconds": round(latency, 3),
                    "chunk_count": chunk_count,
                    "zero_english_required": experiment.prompt_profile in ZERO_ENGLISH_PROFILES,
                    "latin_residue_policy": "record",
                    "latin_residue_count": len(LATIN_LETTER_RE.findall(translation)),
                    "latin_residue_samples": find_latin_residues(translation),
                    "latin_residue_warning": contains_latin_letters(translation),
                    "completed_at": _now(),
                }
                translated_count += 1

            if retry_errors_only:
                repair_results[pid] = item
            else:
                ordered_results.append(item)
            checkpoint_results = current_repair_results()
            state["translations"] = checkpoint_results
            state["errors"] = [x for x in state["errors"] if str(x.get("paragraph_id")) != pid]
            state["statistics"] = _translation_statistics(paragraphs, checkpoint_results, restored_count, state["errors"])
            state["updated_at"] = _now()
            _atomic_json(output_path, state)
    except KeyboardInterrupt:
        final_results = current_repair_results()
        state["translations"] = final_results
        state["statistics"] = _translation_statistics(paragraphs, final_results, restored_count, state["errors"])
        state["status"] = "interrupted"
        state["updated_at"] = _now()
        _atomic_json(output_path, state)
        raise
    finally:
        http_client.close()

    final_results = current_repair_results()
    state["translations"] = final_results
    state["statistics"] = _translation_statistics(paragraphs, final_results, restored_count, state["errors"])
    state["statistics"]["newly_translated_paragraphs"] = translated_count
    state["status"] = (
        "complete"
        if len(final_results) == len(paragraphs) and not state["errors"]
        else "incomplete"
    )
    state["updated_at"] = _now()
    _atomic_json(output_path, state)
    return state


def _translate_source_text(
    *,
    client: Any,
    model: str,
    prompt_profile: str,
    source_text: str,
    previous_context: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    source_chunk_chars: int,
    extra_body: dict[str, Any] | None,
) -> tuple[str, int, float, int]:
    chunks = _split_source_text(source_text, source_chunk_chars)
    translations: list[str] = []
    total_attempts = 0
    total_latency = 0.0
    for chunk_index, chunk in enumerate(chunks):
        chunk_context = previous_context if chunk_index == 0 else ""
        translation, attempts, latency = _call_translation(
            client=client,
            model=model,
            prompt_profile=prompt_profile,
            source_text=chunk,
            previous_context=chunk_context,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            extra_body=extra_body,
        )
        translations.append(translation)
        total_attempts += attempts
        total_latency += latency
    return "\n".join(translations), total_attempts, total_latency, len(chunks)


def _split_source_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    boundary = re.compile(r"(?<=[.;:!?。；：！？])\s+|\n+")
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        matches = list(boundary.finditer(window))
        cut = matches[-1].end() if matches and matches[-1].end() >= max_chars // 2 else max_chars
        chunk = remaining[:cut].strip()
        if not chunk:
            chunk = remaining[:max_chars]
            cut = max_chars
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _build_translation_messages(
    *,
    prompt_profile: str,
    source_text: str,
    previous_context: str,
) -> list[dict[str, str]]:
    """Build an unambiguous request separating reference context from target text."""
    context_section = previous_context.strip() or "（无前文上下文）"
    profile_requirements = PROFILE_TASK_REQUIREMENTS[prompt_profile]
    task = (
        "你正在逐段翻译同一份法律文件。\n"
        "下面的【前文上下文】已经翻译完成，只用于理解语义并保持术语一致。"
        "不得翻译、改写、概括、纠正或输出该部分。\n"
        "你只能翻译【当前待译段落】。\n"
        f"本次翻译要求：{profile_requirements}\n\n"
        "================【前文上下文：仅供参考，禁止输出】================\n"
        f"{context_section}\n"
        "================【前文上下文结束】================\n\n"
        "================【当前待译段落：唯一翻译对象】================\n"
        f"{source_text}\n"
        "================【当前待译段落结束】================\n\n"
        "请严格遵守系统要求，只输出【当前待译段落】的完整中文译文。"
        "不要输出前文上下文、英文原文、标签、说明、注释或致谢。"
    )
    return [
        {"role": "system", "content": PROMPT_PROFILES[prompt_profile]},
        {"role": "user", "content": task},
    ]


def _call_translation(
    *,
    client: Any,
    model: str,
    prompt_profile: str,
    source_text: str,
    previous_context: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    extra_body: dict[str, Any] | None,
) -> tuple[str, int, float]:
    """Call the model once, retrying only max-token truncation.

    ``max_retries`` remains for CLI compatibility, but retries are capped at
    two additional attempts. Non-truncation errors fail immediately. On each
    truncation retry, max_tokens doubles.
    """
    messages = _build_translation_messages(
        prompt_profile=prompt_profile,
        source_text=source_text,
        previous_context=previous_context,
    )
    truncation_retries = min(2, max(0, max_retries - 1))
    started = time.monotonic()

    for attempt_index in range(truncation_retries + 1):
        current_max_tokens = max_tokens * (2 ** attempt_index)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": current_max_tokens,
            "reasoning_effort": "low",
        }
        if extra_body:
            request["extra_body"] = extra_body

        try:
            response = client.chat.completions.create(**request)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Network, server, parsing, and all other failures are not retried.
            raise TranslationError(str(exc)) from exc

        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            if attempt_index < truncation_retries:
                continue
            raise TranslationTruncatedError(
                "translation was truncated by max_tokens "
                f"after {attempt_index + 1} attempts; final max_tokens={current_max_tokens}"
            )

        translation = _clean_translation(choice.message.content)
        if not translation:
            raise TranslationError("model returned an empty translation")

        return translation, attempt_index + 1, time.monotonic() - started

    raise TranslationError("translation failed")

def _clean_translation(value: Any) -> str:
    """Return only the final translation, removing model reasoning wrappers.

    DeepSeek-R1 distill models may emit either a complete ``<think>...</think>``
    block or omit the opening ``<think>`` because it was injected by the chat
    template. In the latter case, everything before the final ``</think>`` is
    reasoning and must not be saved or included in residue statistics.
    """
    text = str(value or "").strip()

    # Some DeepSeek templates inject the opening <think> into the prompt, so
    # generated content can contain only the closing tag. Keep only the text
    # after the last closing tag. This also handles multiple reasoning blocks.
    closing_matches = list(re.finditer(r"</think\s*>", text, re.I))
    if closing_matches:
        text = text[closing_matches[-1].end():].strip()
    else:
        # Handle ordinary complete or unclosed reasoning blocks defensively.
        text = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            text,
            flags=re.S | re.I,
        ).strip()
        text = re.sub(
            r"^\s*<think\b[^>]*>.*$",
            "",
            text,
            flags=re.S | re.I,
        ).strip()

    fenced = re.fullmatch(r"```(?:text|markdown)?\s*(.*?)\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    prefixes = ("翻译：", "译文：", "Translation:")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    return text


def _build_previous_context(
    *,
    paragraphs: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    current_index: int,
    count: int,
    max_chars: int,
) -> str:
    """Format aligned previous paragraphs as read-only bilingual context."""
    if count <= 0 or max_chars <= 0 or not completed:
        return ""

    recent = completed[-count:]
    blocks: list[str] = []
    for position, item in enumerate(recent, start=1):
        source = str(item.get("source", "")).strip()
        target = str(item.get("translation", "")).strip()
        blocks.append(
            f"【前文段落 {position}】\n"
            f"原文（仅供理解，不要翻译或输出）：\n{source}\n"
            f"既有译文（仅供术语一致性，不要输出）：\n{target}"
        )

    # Prefer dropping the oldest whole context blocks over cutting through a block.
    while blocks and len("\n\n".join(blocks)) > max_chars:
        if len(blocks) == 1:
            blocks[0] = blocks[0][-max_chars:]
            break
        blocks.pop(0)
    return "\n\n---------------- 前文段落分隔 ----------------\n\n".join(blocks)

def _translation_statistics(
    paragraphs: list[dict[str, Any]],
    translations: list[dict[str, Any]],
    restored_count: int,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    nonempty = sum(1 for x in translations if str(x.get("translation", "")).strip())
    failed_ids = {
        str(x.get("paragraph_id"))
        for x in (errors or [])
        if isinstance(x, dict) and x.get("paragraph_id") is not None
    }
    return {
        "total_paragraphs": len(paragraphs),
        "translated_paragraphs": len(translations),
        "nonempty_translations": nonempty,
        "empty_translations": len(translations) - nonempty,
        "failed_paragraphs": len(failed_ids),
        "remaining_paragraphs": max(0, len(paragraphs) - len(translations)),
        "restored_paragraphs": restored_count,
    }


def _iter_dataset_documents(input_path: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if input_path.is_file():
        yield input_path, _read_dataset(input_path)
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    for path in sorted(input_path.rglob("*.json")):
        if path.name.endswith("manifest.json"):
            continue
        try:
            data = _read_dataset(path)
        except (ValueError, json.JSONDecodeError):
            continue
        yield path, data


def _read_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "paragraphs" not in data:
        raise ValueError(f"not a structured dataset document: {path}")
    return data


def _source_signature(dataset: dict[str, Any]) -> str:
    payload = {
        "document_id": dataset.get("document_id"),
        "paragraphs": [
            {"paragraph_id": x.get("paragraph_id"), "text": x.get("text", "")}
            for x in dataset.get("paragraphs", [])
        ],
    }
    return _json_hash(payload)


def _experiment_metadata(experiment: TranslationExperiment) -> dict[str, Any]:
    data = asdict(experiment)
    data["system_prompt"] = PROMPT_PROFILES[experiment.prompt_profile]
    data["user_prompt_template"] = USER_PROMPT_PROFILES[experiment.prompt_profile]
    data["system_prompt_hash"] = _text_hash(data["system_prompt"])
    data["user_prompt_hash"] = _text_hash(data["user_prompt_template"])
    data["prompt_hash"] = _json_hash({
        "system": data["system_prompt"],
        "user": data["user_prompt_template"],
    })
    return data


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return result[:180] or "experiment"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _text_hash(text)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, data: Any) -> None:
    atomic_write_json(path, data)