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
from .consistency import (
    aggregate_rendering_tree,
    compare_models,
    compute_final_metrics_tree,
    judge_consistency_tree,
)
from .evaluate import compute_metrics, locate_all_term_translations, locate_term_translations
from .renderings import extract_renderings_with_llm
from .rendering_cleanup import (
    repair_rendering_tree,
    seed_reusable_judgment_tree,
    validate_rendering_tree,
)
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
from .term_memory import build_term_memory_tree
from .controlled_translation import controlled_batch_translate
from .occurrence_validation import validate_term_occurrence_tree
from .translation_evaluation import evaluate_translation_tree


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

    p = sub.add_parser("build-term-memory", help="Build definition-grounded document-level Chinese terminology policies")
    p.add_argument("annotated", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("output", type=Path, help="Term-memory output directory")
    p.add_argument("--aggregated", type=Path, help="Optional clean aggregated renderings used only as candidate evidence")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.add_argument("--max-candidate-variants", type=int, default=12)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-documents", type=int, help="Build memory for the first N selected documents")
    p.add_argument("--document-ids", help="Comma-separated exact document IDs")

    p = sub.add_parser("validate-term-occurrences", help="Disambiguate defined-term uses before glossary or placeholder control")
    p.add_argument("annotated", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("output", type=Path, help="Occurrence-validation JSON or directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-documents", type=int, help="Validate the first N selected documents")
    p.add_argument("--document-ids", help="Comma-separated exact document IDs")

    p = sub.add_parser("controlled-translate", help="Translate with definition-grounded glossary or hard placeholder controls")
    p.add_argument("annotated", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("term_memory", type=Path, help="Term-memory JSON or directory")
    p.add_argument("output", type=Path, help="Controlled translation experiment root")
    p.add_argument("--experiment", required=True)
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--constraint-mode", choices=["placeholder", "glossary"], default="placeholder")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--context-paragraphs", type=int, default=2)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--placeholder-repair-policy",
        choices=["retry_then_fallback", "llm_repair"],
        default="retry_then_fallback",
        help="Safe default regenerates from protected source; llm_repair is legacy ablation.",
    )
    p.add_argument(
        "--placeholder-semantic-verifier",
        choices=["off", "retry", "all"],
        default="retry",
        help="Verify retried placeholder outputs by default; all verifies first-pass outputs too.",
    )
    p.add_argument(
        "--disable-placeholder-quality-gate",
        action="store_true",
        help="Disable deterministic wrapper/restoration checks (not recommended).",
    )
    p.add_argument(
        "--placeholder-repeat-rejection-limit",
        type=int,
        default=2,
        help="Fall back after the same deterministic rejection repeats N times; 0 disables early fallback.",
    )
    p.add_argument(
        "--semantic-verifier-retries",
        type=int,
        default=2,
        help="Retry semantic-verifier request/JSON parsing without regenerating the translation.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--max-documents", type=int, help="Pilot on the first N selected documents")
    p.add_argument("--document-ids", help="Comma-separated exact document IDs for a controlled pilot")
    p.add_argument(
        "--occurrence-validation",
        type=Path,
        help="Optional precomputed occurrence validation. If omitted, the same deterministic validator runs inline.",
    )

    p = sub.add_parser(
        "llm-evaluate-translation",
        aliases=["evaluate-translation"],
        help="Blind LLM judge for baseline versus controlled legal translations",
    )
    p.add_argument("annotated", type=Path, help="Annotated source dataset JSON or directory")
    p.add_argument("baseline", type=Path, help="Baseline translation JSON or experiment directory")
    p.add_argument("controlled", type=Path, help="Controlled translation JSON or experiment directory")
    p.add_argument("output", type=Path, help="Evaluation experiment root")
    p.add_argument("--experiment", required=True, help="Unique evaluation experiment name")
    p.add_argument("--term-memory", type=Path, help="Optional frozen term-memory JSON or directory")
    p.add_argument("--model", default="deepseek-v4-flash", help="Judge model ID")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument(
        "--api-key-env",
        help="Environment variable containing the API key; auto-selects DEEPSEEK_API_KEY for DeepSeek URLs",
    )
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--controlled-label", default="controlled")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--position-control",
        choices=["fixed", "deterministic", "bidirectional"],
        default="deterministic",
        help="Blind A/B order control; bidirectional doubles judge calls and requires agreement for a non-tie winner",
    )
    p.add_argument("--save-raw", action="store_true", help="Store raw judge responses for audit")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-documents", type=int, help="Evaluate the first N selected documents")
    p.add_argument("--document-ids", help="Comma-separated exact document IDs")
    p.add_argument("--max-paragraphs-per-document", type=int, help="Pilot limit within each selected document")
    p.add_argument("--extra-body-json", help="Optional JSON object passed as OpenAI-compatible extra_body")
    p.add_argument(
        "--evaluation-scope",
        choices=["all", "term-bearing", "term-bearing-plus-sample"],
        default="term-bearing-plus-sample",
        help="Low-cost default: evaluate every term-bearing paragraph plus a deterministic sample of other paragraphs",
    )
    p.add_argument(
        "--non-term-sample-rate",
        type=float,
        default=0.1,
        help="Sampling rate for non-term-bearing paragraphs under term-bearing-plus-sample",
    )
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument(
        "--disable-json-mode",
        action="store_true",
        help="Do not request response_format=json_object; normally leave JSON mode enabled",
    )
    p.add_argument(
        "--max-token-escalations",
        type=int,
        default=1,
        help="On truncation, double max_tokens at most this many times; never repeat the same truncated request",
    )

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

    p = sub.add_parser("validate-renderings", help="Validate rendering boundaries and flag suspicious alignments")
    p.add_argument("input", type=Path, help="Rendered review JSON, experiment directory, or data/renderings root")
    p.add_argument("annotated", type=Path, help="Annotated dataset JSON or directory")
    p.add_argument("output", type=Path, help="Validated output JSON or directory")

    p = sub.add_parser("llm-repair-renderings", help="Re-extract only suspicious renderings using a minimal-span LLM prompt")
    p.add_argument("input", type=Path, help="Validated rendering JSON or directory")
    p.add_argument("output", type=Path, help="Repaired output JSON or directory")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="Retry only occurrences previously marked REPAIR_FAILED_RETAINED")
    p.add_argument("--max-output-tokens", type=int, default=1200)

    p = sub.add_parser("metrics", help="Compute DTR, TVR and entropy from completed review JSON")
    p.add_argument("review", type=Path)
    p.add_argument("output", type=Path)

    p = sub.add_parser("aggregate-renderings", help="Aggregate occurrence renderings by document_id + term_id")
    p.add_argument("input", type=Path, help="One rendered review, one experiment directory, or data/renderings root")
    p.add_argument("output", type=Path, help="Aggregated JSON or output directory")
    p.add_argument("--max-contexts-per-variant", type=int, default=3)

    p = sub.add_parser("llm-judge-consistency", help="Judge semantic/legal consistency of distinct renderings")
    p.add_argument("input", type=Path, help="Aggregated JSON or aggregated directory")
    p.add_argument("output", type=Path, help="Judgment JSON or output directory")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--base-url", default="http://10.12.143.51:8000/v1")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("reuse-judgments", help="Reuse old judgments only for groups unchanged after rendering cleanup")
    p.add_argument("old_aggregated", type=Path)
    p.add_argument("old_judgments", type=Path)
    p.add_argument("new_aggregated", type=Path)
    p.add_argument("output", type=Path, help="Seed judgment JSON or directory; changed groups are omitted")

    p = sub.add_parser("final-metrics", help="Combine formal metrics and legal-semantic judgments")
    p.add_argument("aggregated", type=Path, help="Aggregated JSON or directory")
    p.add_argument("judgments", type=Path, help="Judgment JSON or directory")
    p.add_argument("output", type=Path, help="Metrics JSON or directory")

    p = sub.add_parser("compare-models", help="Create a cross-model comparison JSON and CSV")
    p.add_argument("input", type=Path, help="Final metrics JSON or metrics directory")
    p.add_argument("output", type=Path, help="Comparison JSON")
    p.add_argument("--csv", type=Path, dest="csv_output")

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
    elif args.command == "build-term-memory":
        result = build_term_memory_tree(
            args.annotated,
            args.output,
            model=args.model,
            base_url=args.base_url,
            aggregated=args.aggregated,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            confidence_threshold=args.confidence_threshold,
            max_candidate_variants=args.max_candidate_variants,
            overwrite=args.overwrite,
            max_documents=args.max_documents,
            document_ids={x.strip() for x in args.document_ids.split(",") if x.strip()} if args.document_ids else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "validate-term-occurrences":
        result = validate_term_occurrence_tree(
            args.annotated,
            args.output,
            overwrite=args.overwrite,
            max_documents=args.max_documents,
            document_ids={x.strip() for x in args.document_ids.split(",") if x.strip()} if args.document_ids else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "controlled-translate":
        result = controlled_batch_translate(
            args.annotated,
            args.term_memory,
            args.output,
            experiment_name=args.experiment,
            model=args.model,
            base_url=args.base_url,
            constraint_mode=args.constraint_mode,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_paragraphs=args.context_paragraphs,
            max_context_chars=args.max_context_chars,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            workers=args.workers,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
            max_documents=args.max_documents,
            document_ids={x.strip() for x in args.document_ids.split(",") if x.strip()} if args.document_ids else None,
            occurrence_validation=args.occurrence_validation,
            placeholder_repair_policy=args.placeholder_repair_policy,
            placeholder_quality_gate=not args.disable_placeholder_quality_gate,
            placeholder_semantic_verifier=args.placeholder_semantic_verifier,
            placeholder_repeat_rejection_limit=args.placeholder_repeat_rejection_limit,
            semantic_verifier_retries=args.semantic_verifier_retries,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command in {"llm-evaluate-translation", "evaluate-translation"}:
        result = evaluate_translation_tree(
            args.annotated,
            args.baseline,
            args.controlled,
            args.output,
            experiment_name=args.experiment,
            model=args.model,
            base_url=args.base_url,
            term_memory=args.term_memory,
            api_key_env=args.api_key_env,
            baseline_label=args.baseline_label,
            controlled_label=args.controlled_label,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            workers=args.workers,
            position_control=args.position_control,
            save_raw=args.save_raw,
            overwrite=args.overwrite,
            max_documents=args.max_documents,
            document_ids={x.strip() for x in args.document_ids.split(",") if x.strip()} if args.document_ids else None,
            max_paragraphs_per_document=args.max_paragraphs_per_document,
            extra_body=json.loads(args.extra_body_json) if args.extra_body_json else None,
            evaluation_scope=args.evaluation_scope,
            non_term_sample_rate=args.non_term_sample_rate,
            sample_seed=args.sample_seed,
            json_mode=not args.disable_json_mode,
            max_token_escalations=args.max_token_escalations,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
    elif args.command == "validate-renderings":
        result = validate_rendering_tree(args.input, args.annotated, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "llm-repair-renderings":
        result = repair_rendering_tree(
            args.input,
            args.output,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            checkpoint_path=args.checkpoint,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
            max_output_tokens=args.max_output_tokens,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "metrics":
        compute_metrics(args.review, args.output)
    elif args.command == "aggregate-renderings":
        result = aggregate_rendering_tree(
            args.input,
            args.output,
            max_contexts_per_variant=args.max_contexts_per_variant,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "llm-judge-consistency":
        result = judge_consistency_tree(
            args.input,
            args.output,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            checkpoint_path=args.checkpoint,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "reuse-judgments":
        result = seed_reusable_judgment_tree(
            args.old_aggregated, args.old_judgments, args.new_aggregated, args.output
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "final-metrics":
        result = compute_final_metrics_tree(args.aggregated, args.judgments, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "compare-models":
        result = compare_models(args.input, args.output, output_csv=args.csv_output)
        print(json.dumps(result, ensure_ascii=False, indent=2))


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
