from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from common import load_config, resolve

ABLATIONS = ["full", "no-semantic-verifier", "no-retry", "no-deterministic-quality-gate", "legacy-llm-repair"]


def q(value: object) -> str:
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-cost LLM-judge commands for closed-loop ablations.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--model-key", default="gpt-oss-20b")
    args = parser.parse_args()

    config, root = load_config(args.config)
    paths, judge = config["paths"], config["judge"]
    output_root = resolve(root, paths["output_root"]) / "ablations"
    controlled_root = output_root / "translations_controlled"
    eval_root = output_root / "evaluations"
    logs = output_root / "logs"
    annotated = resolve(root, paths["annotated"])
    baseline = resolve(root, paths["baseline"])
    term_memory = resolve(root, paths["term_memory"])
    script = output_root / "run_ablation_judges.sh"

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"mkdir -p {q(eval_root)} {q(logs)}", ""]
    for name in ABLATIONS:
        controlled = controlled_root / f"{args.model_key}-ablation-{name}"
        experiment = f"{args.model_key}-ablation-{name}-judge"
        log = logs / f"{experiment}.log"
        parts = [
            "PYTHONUNBUFFERED=1 nohup legal-tc llm-evaluate-translation",
            q(annotated), q(baseline), q(controlled), q(eval_root),
            "--experiment", q(experiment),
            "--term-memory", q(term_memory),
            "--model", q(judge["model"]),
            "--base-url", q(judge["base_url"]),
            "--controlled-label", q(name),
            "--position-control", "deterministic",
            "--evaluation-scope", "term-bearing",
            "--extra-body-json", q(judge.get("extra_body_json", '{"thinking":{"type":"disabled"}}')),
            "--max-tokens", str(judge.get("max_tokens", 1200)),
            "--max-retries", str(judge.get("max_retries", 2)),
            "--max-token-escalations", str(judge.get("max_token_escalations", 1)),
            "--workers", str(judge.get("workers", 16)),
            "--timeout", str(judge.get("timeout", 300)),
            ">", q(log), "2>&1",
        ]
        lines.append(" \\\n  ".join(parts))
        lines.append(f"echo {q(experiment + ' complete')}")
        lines.append("")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    print(f"Generated: {script}")


if __name__ == "__main__":
    main()
