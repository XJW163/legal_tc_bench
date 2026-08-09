from __future__ import annotations

import argparse
from pathlib import Path

from common import atomic_json, check_legal_tc, experiment_registry, load_config, p, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    annotated = p(root, config["paths"]["annotated"])
    reviews_root = p(root, config["paths"]["reviews_root"])
    logs_root = p(root, config["paths"]["results_root"]) / "logs"
    manifest = []
    for item in experiment_registry(config, root, include_missing=True):
        source = Path(item["translation_dir"])
        destination = reviews_root / item["experiment_id"]
        if not source.is_dir():
            if args.skip_missing:
                manifest.append({"experiment_id": item["experiment_id"], "status": "missing", "source": str(source)})
                continue
            raise FileNotFoundError(source)
        run(
            [legal_tc, "make-reviews", annotated, source, destination],
            cwd=root,
            log_path=logs_root / "04_build_reviews.log",
        )
        manifest.append({"experiment_id": item["experiment_id"], "status": "ok", "source": str(source), "output": str(destination)})
    atomic_json(reviews_root / "final_review_registry.json", {"experiments": manifest})


if __name__ == "__main__":
    main()
