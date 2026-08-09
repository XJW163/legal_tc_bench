from __future__ import annotations

import argparse
from pathlib import Path

from common import advertised_models, atomic_json, check_legal_tc, experiment_registry, load_config, p, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    judge = config["judge"]
    if not args.skip_server_check:
        available = advertised_models(str(judge["base_url"]))
        if str(judge["model"]) not in available:
            raise RuntimeError(f"judge endpoint advertises {available}, expected {judge['model']}")

    reviews_root = p(root, config["paths"]["reviews_root"])
    renderings_root = p(root, config["paths"]["renderings_root"])
    logs_root = p(root, config["paths"]["results_root"]) / "logs"
    manifest = []
    for item in experiment_registry(config, root, include_missing=True):
        source = reviews_root / item["experiment_id"]
        destination = renderings_root / item["experiment_id"]
        if not source.is_dir():
            if args.skip_missing:
                manifest.append({"experiment_id": item["experiment_id"], "status": "missing", "source": str(source)})
                continue
            raise FileNotFoundError(source)
        cmd = [
            legal_tc,
            "llm-extract-renderings",
            source,
            destination,
            "--model",
            str(judge["model"]),
            "--base-url",
            str(judge["base_url"]),
            "--batch-size",
            str(judge["rendering_batch_size"]),
            "--workers",
            str(judge["rendering_workers"]),
            "--max-retries",
            str(judge["rendering_max_retries"]),
            "--timeout",
            str(judge["rendering_timeout_seconds"]),
        ]
        run(cmd, cwd=root, log_path=logs_root / "05_extract_renderings.log")
        manifest.append({"experiment_id": item["experiment_id"], "status": "ok", "output": str(destination)})
    atomic_json(renderings_root / "final_rendering_registry.json", {"experiments": manifest})


if __name__ == "__main__":
    main()
