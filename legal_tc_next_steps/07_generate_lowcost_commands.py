from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from common import load_config, resolve


def q(value: object) -> str:
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-sample DeepSeek judge commands for the remaining translator models.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--matrix", default="model_matrix.json")
    parser.add_argument("--non-term-sample-rate", type=float, default=0.10)
    parser.add_argument("--sample-seed", type=int, default=2026)
    args = parser.parse_args()

    config, root = load_config(args.config)
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    judge = config["judge"]
    annotated = resolve(root, paths["annotated"])
    term_memory = resolve(root, paths["term_memory"])
    evaluation_root = resolve(root, paths["evaluation_root"])
    logs = root / "data/logs"
    output = resolve(root, paths["output_root"]) / "07_run_remaining_models_lowcost.sh"

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "mkdir -p " + q(logs) + " " + q(evaluation_root), ""]
    generated = 0
    for model in matrix.get("models") or []:
        if not model.get("enabled", True):
            continue
        key = str(model["key"])
        baseline = resolve(root, model["baseline"])
        for method in ("glossary", "placeholder"):
            controlled = resolve(root, model[method])
            experiment = f"{key}-{method}-deepseek-v4-flash-lowcost"
            log = logs / f"{experiment}.log"
            command = [
                "PYTHONUNBUFFERED=1 nohup legal-tc llm-evaluate-translation",
                q(annotated),
                q(baseline),
                q(controlled),
                q(evaluation_root),
                "--experiment", q(experiment),
                "--term-memory", q(term_memory),
                "--model", q(judge["model"]),
                "--base-url", q(judge["base_url"]),
                "--controlled-label", q(method),
                "--position-control", "deterministic",
                "--evaluation-scope", "term-bearing-plus-sample",
                "--non-term-sample-rate", str(args.non_term_sample_rate),
                "--sample-seed", str(args.sample_seed),
                "--extra-body-json", q(judge.get("extra_body_json", '{"thinking":{"type":"disabled"}}')),
                "--max-tokens", str(judge.get("max_tokens", 1200)),
                "--max-retries", str(judge.get("max_retries", 2)),
                "--max-token-escalations", str(judge.get("max_token_escalations", 1)),
                "--workers", str(judge.get("workers", 16)),
                "--timeout", str(judge.get("timeout", 300)),
                ">", q(log), "2>&1",
            ]
            lines.append(" \\\n  ".join(command))
            lines.append(f"echo {q(experiment + ' complete')}")
            lines.append("")
            generated += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.chmod(0o755)
    print(f"Generated {generated} commands: {output}")
    print("Review all directory names before executing the shell script.")


if __name__ == "__main__":
    main()
