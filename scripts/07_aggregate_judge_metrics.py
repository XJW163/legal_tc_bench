from __future__ import annotations

import argparse

from common import advertised_models, check_legal_tc, load_config, p, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    judge = config["judge"]
    if not args.skip_server_check:
        available = advertised_models(str(judge["base_url"]))
        if str(judge["model"]) not in available:
            raise RuntimeError(f"judge endpoint advertises {available}, expected {judge['model']}")

    paths = config["paths"]
    clean = p(root, paths["renderings_clean_root"])
    aggregated = p(root, paths["aggregated_root"])
    judgments = p(root, paths["judgments_root"])
    metrics = p(root, paths["final_metrics_root"])
    results = p(root, paths["results_root"])
    logs = results / "logs"

    run([legal_tc, "aggregate-renderings", clean, aggregated], cwd=root, log_path=logs / "07_aggregate.log")
    run(
        [
            legal_tc,
            "llm-judge-consistency",
            aggregated,
            judgments,
            "--model",
            str(judge["model"]),
            "--base-url",
            str(judge["base_url"]),
            "--batch-size",
            str(judge["consistency_batch_size"]),
            "--confidence-threshold",
            str(judge["consistency_confidence_threshold"]),
            "--max-retries",
            str(judge["consistency_max_retries"]),
            "--timeout",
            str(judge["consistency_timeout_seconds"]),
        ],
        cwd=root,
        log_path=logs / "07_judge_consistency.log",
    )
    run([legal_tc, "final-metrics", aggregated, judgments, metrics], cwd=root, log_path=logs / "07_final_metrics.log")
    run(
        [legal_tc, "compare-models", metrics, results / "model_comparison.json", "--csv", results / "model_comparison.csv"],
        cwd=root,
        log_path=logs / "07_compare_models.log",
    )


if __name__ == "__main__":
    main()
