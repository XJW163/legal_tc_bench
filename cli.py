from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .annotations import apply_term_review, export_term_review
from .llm_annotations import annotate_terms_with_llm
from .build import build_document
from .evaluate import compute_metrics, locate_all_term_translations, locate_term_translations
from .renderings import extract_renderings_with_llm
from .sec import SecClient, discover_exhibits, parse_master_index, quarter_index_url, save_exhibit
from .translate import (
    PROMPT_PROFILES,
    batch_translate,
    check_openai_compatible_server,
    run_translation_matrix,
    translate_paragraphs,
    validate_translation_directory,
)
from .text import write_clean_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="legal-tc")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-sec", help="Sample and download material-contract exhibits from SEC EDGAR")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarters", default="1,2,3,4")
    p.add_argument("--max-contracts", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/raw"))
    p.add_argument("--forms", default="8-K,10-K,10-Q,S-1,S-4,20-F,6-K")
    p.add_argument("--rps", type=float, default=5.0)

    p = sub.add_parser("parse", help="Convert one SEC/HTML contract into readable UTF-8 text")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)

    p = sub.add_parser("build", help="Parse downloaded contracts and extract defined terms")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/dataset"))
    p.add_argument("--min-occurrences", type=int, default=3)

    p = sub.add_parser("export-term-review", help="Export candidate terms to a human-review CSV")
    p.add_argument("input", type=Path, help="One dataset JSON or a dataset directory")
    p.add_argument("output", type=Path, help="UTF-8 CSV for Excel/WPS review")

    p = sub.add_parser("llm-annotate-terms", help="Annotate candidate legal terms with an LLM")
    p.add_argument("input", type=Path, help="One dataset JSON or a dataset directory")
    p.add_argument("output", type=Path, help="Completed review CSV")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=3)

    p = sub.add_parser("apply-term-review", help="Validate and apply reviewed term annotations")
    p.add_argument("input", type=Path, help="One dataset JSON or a dataset directory")
    p.add_argument("review", type=Path, help="Completed review CSV")
    p.add_argument("output", type=Path, help="Output JSON or directory")
    p.add_argument("--accept-suggestions", action="store_true", help="Fill blank manual fields from heuristic suggestions")
    p.add_argument("--allow-unreviewed", action="store_true", help="Do not fail when a candidate has no review row")

    p = sub.add_parser("translate", help="Translate one dataset JSON with resumable vLLM/OpenAI-compatible chat completions")
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default="baseline-v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--source-chunk-chars", type=int, default=12000)
    p.add_argument("--context-paragraphs", type=int, default=2)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--extra-body-json", help="Optional JSON object passed to chat.completions, e.g. Qwen thinking controls")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="Retry only paragraph IDs recorded in the existing output file's errors list",
    )

    p = sub.add_parser("batch-translate", help="Translate a dataset directory into one isolated model experiment")
    p.add_argument("input", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("output", type=Path, help="Translation experiment root, e.g. data/translations")
    p.add_argument("--experiment", required=True, help="Unique experiment name used as the output subdirectory")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default="baseline-v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--source-chunk-chars", type=int, default=12000)
    p.add_argument("--context-paragraphs", type=int, default=2, help="Include N previous aligned source/translation pairs as read-only context")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--extra-body-json", help="Optional JSON object passed to chat.completions")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Number of documents translated concurrently; each document remains sequential")

    p = sub.add_parser("translate-matrix", help="Run multiple model/prompt experiments from a JSON configuration")
    p.add_argument("config", type=Path)
    p.add_argument("--input", type=Path, help="Override config input")
    p.add_argument("--output", type=Path, help="Override config output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p = sub.add_parser("validate-translations", help="Check document and paragraph coverage for one experiment")
    p.add_argument("input", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("experiment", type=Path, help="One translation experiment directory")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("check-models", help="List model IDs advertised by an OpenAI-compatible server")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--timeout", type=float, default=30.0)

    p = sub.add_parser("make-review", help="Create one occurrence-level terminology review JSON")
    p.add_argument("dataset", type=Path)
    p.add_argument("translation", type=Path)
    p.add_argument("output", type=Path)

    p = sub.add_parser("make-reviews", help="Create occurrence-level review JSONs for matching directories")
    p.add_argument("dataset", type=Path, help="Annotated dataset directory")
    p.add_argument("translation", type=Path, help="Translation JSON directory")
    p.add_argument("output", type=Path, help="Output review directory")

    p = sub.add_parser("llm-extract-renderings", help="Extract exact Chinese term renderings with an LLM")
    p.add_argument("input", type=Path, help="One review JSON or a review directory")
    p.add_argument("output", type=Path, help="Output JSON or directory")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="Concurrent LLM requests")
    p.add_argument("--max-output-tokens", type=int, default=384)
    p.add_argument("--max-source-chars", type=int, default=700)
    p.add_argument("--max-target-chars", type=int, default=2200)
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate remaining token requirements without calling the LLM",
    )

    p = sub.add_parser("metrics", help="Compute DTR, TVR and entropy from completed review JSON")
    p.add_argument("review", type=Path)
    p.add_argument("output", type=Path)

    args = parser.parse_args()
    if args.command == "download-sec":
        download_sec(args)
    elif args.command == "parse":
        paragraphs = write_clean_text(args.input, args.output)
        print(f"wrote {len(paragraphs)} paragraphs to {args.output}")
    elif args.command == "build":
        build_all(args)
    elif args.command == "export-term-review":
        count = export_term_review(args.input, args.output)
        print(f"wrote {count} candidate terms to {args.output}")
    elif args.command == "llm-annotate-terms":
        result = annotate_terms_with_llm(
            args.input,
            args.output,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            max_retries=args.max_retries,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "apply-term-review":
        result = apply_term_review(
            args.input,
            args.review,
            args.output,
            accept_suggestions=args.accept_suggestions,
            strict=not args.allow_unreviewed,
        )
        print(json.dumps(result["totals"], ensure_ascii=False, indent=2))
    elif args.command == "translate":
        result = translate_paragraphs(
            args.dataset,
            args.output,
            args.model,
            args.base_url,
            prompt_profile=args.prompt_profile,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            source_chunk_chars=args.source_chunk_chars,
            context_paragraphs=args.context_paragraphs,
            max_context_chars=args.max_context_chars,
            extra_body=json.loads(args.extra_body_json) if args.extra_body_json else None,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            overwrite=args.overwrite,
            retry_errors_only=args.retry_errors_only,
        )
        print(json.dumps({"status": result["status"], "statistics": result["statistics"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    elif args.command == "batch-translate":
        result = batch_translate(
            args.input,
            args.output,
            experiment_name=args.experiment,
            model=args.model,
            base_url=args.base_url,
            prompt_profile=args.prompt_profile,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            source_chunk_chars=args.source_chunk_chars,
            context_paragraphs=args.context_paragraphs,
            max_context_chars=args.max_context_chars,
            extra_body=json.loads(args.extra_body_json) if args.extra_body_json else None,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
            workers=args.workers,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "translate-matrix":
        result = run_translation_matrix(
            args.config,
            input_override=args.input,
            output_override=args.output,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "validate-translations":
        result = validate_translation_directory(args.input, args.experiment, output_path=args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "check-models":
        models = check_openai_compatible_server(base_url=args.base_url, timeout_seconds=args.timeout)
        print(json.dumps({"base_url": args.base_url, "models": models}, ensure_ascii=False, indent=2))
    elif args.command == "make-review":
        count = locate_term_translations(args.dataset, args.translation, args.output)
        print(f"wrote {count} occurrences to {args.output}")
    elif args.command == "make-reviews":
        result = locate_all_term_translations(args.dataset, args.translation, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "llm-extract-renderings":
        result = extract_renderings_with_llm(
            args.input,
            args.output,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            workers=args.workers,
            confidence_threshold=args.confidence_threshold,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            checkpoint_path=args.checkpoint,
            overwrite=args.overwrite,
            max_output_tokens=args.max_output_tokens,
            max_source_chars=args.max_source_chars,
            max_target_chars=args.max_target_chars,
            estimate_only=args.estimate_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "metrics":
        compute_metrics(args.review, args.output)


def download_sec(args: argparse.Namespace) -> None:
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise SystemExit('Set SEC_USER_AGENT, e.g. "Jiwei Xu xujiwei@gmail.com"')
    client = SecClient(user_agent, requests_per_second=args.rps)
    forms = {x.strip().upper() for x in args.forms.split(",")}
    filings = []
    cache = args.output.parent / "indexes"
    for q in [int(x) for x in args.quarters.split(",")]:
        url = quarter_index_url(args.year, q)
        idx = client.download(url, cache / f"{args.year}_QTR{q}_master.idx")
        filings.extend(f for f in parse_master_index(idx.read_text(encoding="latin-1")) if f.form.upper() in forms)
    random.Random(args.seed).shuffle(filings)

    manifest = []
    seen_urls = set()
    progress = tqdm(total=args.max_contracts, desc="contracts")
    for exhibit in discover_exhibits(client, filings):
        if exhibit.url in seen_urls:
            continue
        seen_urls.add(exhibit.url)
        try:
            document, metadata = save_exhibit(client, exhibit, args.output)
            clean_path = document.with_suffix(document.suffix + ".txt")
            write_clean_text(document, clean_path)
            manifest.append({**asdict(exhibit), "local_path": str(document), "metadata_path": str(metadata), "clean_text_path": str(clean_path)})
            progress.update(1)
        except Exception as exc:
            tqdm.write(f"skip {exhibit.url}: {exc}")
        if len(manifest) >= args.max_contracts:
            break
    progress.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_all(args: argparse.Namespace) -> None:
    candidates = [p for p in args.raw.rglob("*") if p.is_file() and not p.name.endswith((".json", ".metadata.json"))]
    summary = []
    for path in tqdm(candidates, desc="build"):
        metadata = path.with_name(path.name + ".metadata.json")
        output = args.output / f"{path.parent.name}_{path.stem}.json"
        try:
            doc = build_document(path, metadata, output, args.min_occurrences)
            if doc["statistics"]["term_count"] > 0:
                summary.append({"document_id": doc["document_id"], **doc["statistics"], "path": str(output)})
            else:
                output.unlink(missing_ok=True)
        except Exception as exc:
            tqdm.write(f"skip {path}: {exc}")
    (args.output / "dataset_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
