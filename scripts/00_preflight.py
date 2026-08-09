from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    advertised_models,
    atomic_json,
    check_legal_tc,
    experiment_registry,
    load_config,
    p,
)


def count_document_jsons(directory: Path) -> int:
    count = 0
    if not directory.is_dir():
        return 0
    for path in directory.rglob("*.json"):
        if path.name.endswith(("manifest.json", ".audit.json", ".checkpoint.json")):
            continue
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--check-server", help="model key to check against its configured endpoint")
    args = parser.parse_args()

    config, root = load_config(args.config)
    legal_tc = check_legal_tc()
    paths = {name: p(root, value) for name, value in config["paths"].items()}
    required = ["annotated", "term_memory", "occurrence_validation"]
    path_status = {
        name: {"path": str(paths[name]), "exists": paths[name].exists(), "is_dir": paths[name].is_dir()}
        for name in required
    }

    experiments = []
    for item in experiment_registry(config, root, include_missing=True):
        directory = Path(item["translation_dir"])
        experiments.append({
            **{k: v for k, v in item.items() if k != "translation_dir"},
            "translation_dir": str(directory),
            "exists": directory.is_dir(),
            "json_file_count": count_document_jsons(directory),
        })

    result = {
        "project_root": str(root),
        "legal_tc": legal_tc,
        "paths": path_status,
        "experiments": experiments,
    }
    if args.check_server:
        model = next(x for x in config["models"] if x["key"] == args.check_server)
        try:
            result["server"] = {
                "model_key": args.check_server,
                "base_url": model["base_url"],
                "advertised_models": advertised_models(model["base_url"]),
                "ok": True,
            }
        except Exception as exc:
            result["server"] = {"model_key": args.check_server, "ok": False, "error": str(exc)}

    output = p(root, config["paths"]["results_root"]) / "preflight.json"
    atomic_json(output, result)
    print(f"wrote {output}")

    missing = [name for name, status in path_status.items() if not status["exists"]]
    if missing:
        raise SystemExit("missing required path(s): " + ", ".join(missing))


if __name__ == "__main__":
    main()
