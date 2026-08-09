#!/usr/bin/env python3
"""Evaluate one or more translation experiment directories with the LegalTC pipeline.

Example:
  python run_evaluation.py \
    --experiment gpt-oss-20b-baseline=data/translations/gpt-oss-20b-zero-english-v2 \
    --experiment gpt-oss-20b-glossary=data/translations_controlled/gpt-oss-20b-glossary-final \
    --experiment gpt-oss-20b-placeholder=data/translations_controlled/gpt-oss-20b-placeholder-final \
    --judge-model gpt-oss-120b \
    --judge-base-url http://10.12.143.51:8000/v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def advertised_models(base_url: str, timeout: float = 10.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot query {url}: {exc}") from exc
    return [str(x.get("id")) for x in payload.get("data", []) if x.get("id")]


def parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=TRANSLATION_DIRECTORY")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise argparse.ArgumentTypeError("experiment name is empty")
    return name, path


def run(cmd: list[str | Path], log_path: Path, *, cwd: Path) -> None:
    argv = [str(x) for x in cmd]
    print("+", " ".join(argv), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n+ " + " ".join(argv) + "\n")
        handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, argv)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reviews -> rendering extraction -> validation/repair -> aggregation -> judging -> metrics."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        type=parse_experiment,
        metavar="NAME=DIR",
        help="Repeat once for every baseline/glossary/placeholder translation directory",
    )
    parser.add_argument("--annotated", default="data/annotated")
    parser.add_argument("--work-root", default="data/final_evaluation")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--extract-workers", type=int, default=2)
    parser.add_argument("--repair-batch-size", type=int, default=2)
    parser.add_argument("--judge-batch-size", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--stop-after",
        choices=["reviews", "extract", "repair", "aggregate", "judge", "metrics"],
        default="metrics",
    )
    args = parser.parse_args()

    legal_tc = shutil.which("legal-tc")
    if not legal_tc:
        raise SystemExit("legal-tc is not on PATH; activate .venv and run pip install -e .")

    project_root = Path.cwd()
    annotated = Path(args.annotated)
    if not annotated.is_dir():
        raise SystemExit(f"missing annotated data: {annotated}")

    experiments: list[tuple[str, Path]] = args.experiment
    seen: set[str] = set()
    for name, source in experiments:
        if name in seen:
            raise SystemExit(f"duplicate experiment name: {name}")
        seen.add(name)
        if not source.is_dir():
            raise SystemExit(f"missing translation directory for {name}: {source}")

    if not args.skip_server_check:
        available = advertised_models(args.judge_base_url)
        print("judge endpoint models:", available)
        if args.judge_model not in available:
            raise SystemExit(
                f"judge endpoint advertises {available}, not {args.judge_model!r}"
            )

    root = Path(args.work_root)
    reviews_root = root / "reviews"
    renderings_root = root / "renderings"
    validated_root = root / "renderings_validated"
    clean_root = root / "renderings_clean"
    aggregated_root = root / "aggregated"
    judgments_root = root / "judgments"
    metrics_root = root / "metrics"
    results_root = root / "results"
    logs_root = root / "logs"

    if args.overwrite and root.exists():
        shutil.rmtree(root)

    registry: list[dict[str, str]] = []
    stage_order = ["reviews", "extract", "repair", "aggregate", "judge", "metrics"]
    stop_index = stage_order.index(args.stop_after)

    # Per-experiment stages.
    for name, source in experiments:
        print(f"\n=== {name} ===")
        reviews = reviews_root / name
        renderings = renderings_root / name
        validated = validated_root / name
        clean = clean_root / name

        run(
            [legal_tc, "make-reviews", annotated, source, reviews],
            logs_root / f"{name}.01_reviews.log",
            cwd=project_root,
        )
        if stop_index == 0:
            registry.append({"experiment": name, "translation": str(source), "reviews": str(reviews)})
            continue

        run(
            [
                legal_tc,
                "llm-extract-renderings",
                reviews,
                renderings,
                "--model",
                args.judge_model,
                "--base-url",
                args.judge_base_url,
                "--batch-size",
                str(args.extract_batch_size),
                "--workers",
                str(args.extract_workers),
                "--max-retries",
                str(args.max_retries),
                "--timeout",
                str(args.timeout),
            ],
            logs_root / f"{name}.02_extract.log",
            cwd=project_root,
        )
        if stop_index == 1:
            registry.append({"experiment": name, "renderings": str(renderings)})
            continue

        run(
            [legal_tc, "validate-renderings", renderings, annotated, validated],
            logs_root / f"{name}.03_validate.log",
            cwd=project_root,
        )
        repair_cmd: list[str | Path] = [
            legal_tc,
            "llm-repair-renderings",
            validated,
            clean,
            "--model",
            args.judge_model,
            "--base-url",
            args.judge_base_url,
            "--batch-size",
            str(args.repair_batch_size),
            "--max-retries",
            str(args.max_retries),
            "--timeout",
            str(args.timeout),
        ]
        if args.retry_failed:
            repair_cmd.append("--retry-failed")
        run(repair_cmd, logs_root / f"{name}.04_repair.log", cwd=project_root)
        registry.append(
            {
                "experiment": name,
                "translation": str(source),
                "reviews": str(reviews),
                "renderings": str(renderings),
                "validated": str(validated),
                "clean": str(clean),
            }
        )

    atomic_json(root / "experiment_registry.json", {"experiments": registry})
    if stop_index <= 2:
        print(f"\nstopped after {args.stop_after}; registry: {root / 'experiment_registry.json'}")
        return

    # Tree-wide stages over only this evaluation root.
    run(
        [legal_tc, "aggregate-renderings", clean_root, aggregated_root],
        logs_root / "05_aggregate.log",
        cwd=project_root,
    )
    if stop_index == 3:
        return

    run(
        [
            legal_tc,
            "llm-judge-consistency",
            aggregated_root,
            judgments_root,
            "--model",
            args.judge_model,
            "--base-url",
            args.judge_base_url,
            "--batch-size",
            str(args.judge_batch_size),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--max-retries",
            str(args.max_retries),
            "--timeout",
            str(args.timeout),
        ],
        logs_root / "06_judge.log",
        cwd=project_root,
    )
    if stop_index == 4:
        return

    run(
        [legal_tc, "final-metrics", aggregated_root, judgments_root, metrics_root],
        logs_root / "07_metrics.log",
        cwd=project_root,
    )
    run(
        [
            legal_tc,
            "compare-models",
            metrics_root,
            results_root / "model_comparison.json",
            "--csv",
            results_root / "model_comparison.csv",
        ],
        logs_root / "08_compare.log",
        cwd=project_root,
    )
    print("\nEvaluation complete")
    print("metrics:", metrics_root)
    print("comparison:", results_root / "model_comparison.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
