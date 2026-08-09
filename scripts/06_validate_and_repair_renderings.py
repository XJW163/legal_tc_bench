from __future__ import annotations

import argparse
from pathlib import Path

from common import advertised_models, atomic_json, check_legal_tc, experiment_registry, load_config, p, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    judge = config["judge"]
    if not args.skip_server_check:
        available = advertised_models(str(judge["base_url"]))
        if str(judge["model"]) not in available:
            raise RuntimeError(f"judge endpoint advertises {available}, expected {judge['model']}")

    annotated = p(root, config["paths"]["annotated"])
    renderings_root = p(root, config["paths"]["renderings_root"])
    validated_root = p(root, config["paths"]["renderings_validated_root"])
    clean_root = p(root, config["paths"]["renderings_clean_root"])
    logs_root = p(root, config["paths"]["results_root"]) / "logs"
    manifest = []

    for item in experiment_registry(config, root, include_missing=True):
        source = renderings_root / item["experiment_id"]
        validated = validated_root / item["experiment_id"]
        clean = clean_root / item["experiment_id"]
        if not source.is_dir():
            if args.skip_missing:
                manifest.append({"experiment_id": item["experiment_id"], "status": "missing", "source": str(source)})
                continue
            raise FileNotFoundError(source)
        run([legal_tc, "validate-renderings", source, annotated, validated], cwd=root, log_path=logs_root / "06_validate_renderings.log")
        cmd = [
            legal_tc,
            "llm-repair-renderings",
            validated,
            clean,
            "--model",
            str(judge["model"]),
            "--base-url",
            str(judge["base_url"]),
            "--batch-size",
            str(judge["repair_batch_size"]),
            "--max-retries",
            str(judge["repair_max_retries"]),
            "--timeout",
            str(judge["repair_timeout_seconds"]),
        ]
        if args.retry_failed:
            cmd.append("--retry-failed")
        run(cmd, cwd=root, log_path=logs_root / "06_repair_renderings.log")
        manifest.append({"experiment_id": item["experiment_id"], "status": "ok", "validated": str(validated), "clean": str(clean)})
    atomic_json(clean_root / "final_clean_registry.json", {"experiments": manifest})


if __name__ == "__main__":
    main()
