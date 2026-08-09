from __future__ import annotations

import argparse
import copy
from pathlib import Path

from common import load_config, load_json, resolve, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated gpt-oss-20b config for the project's DTR/SLCR/LIR pipeline.")
    parser.add_argument("--config", default="analysis_config.json")
    parser.add_argument("--base-config", default="config/final_experiment.json")
    args = parser.parse_args()

    analysis, root = load_config(args.config)
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = (root / base_path).resolve()
    base = copy.deepcopy(load_json(base_path))
    paths = analysis["paths"]
    output_root = resolve(root, paths["output_root"]) / "semantic_term_metrics"
    model = next((x for x in base.get("models", []) if x.get("key") == "gpt-oss-20b"), None)
    if model is None:
        raise ValueError("gpt-oss-20b missing from base config")
    model = copy.deepcopy(model)
    model["baseline_candidates"] = [Path(paths["baseline"]).name]
    base["project_root"] = str(root)
    base["models"] = [model]
    base["paths"].update({
        "annotated": str(resolve(root, paths["annotated"])),
        "term_memory": str(resolve(root, paths["term_memory"])),
        "baseline_root": str(resolve(root, paths["baseline"]).parent),
        "controlled_root": str(resolve(root, paths["glossary"]).parent),
        "reviews_root": str(output_root / "reviews"),
        "renderings_root": str(output_root / "renderings"),
        "renderings_validated_root": str(output_root / "renderings_validated"),
        "renderings_clean_root": str(output_root / "renderings_clean"),
        "aggregated_root": str(output_root / "aggregated"),
        "judgments_root": str(output_root / "judgments"),
        "final_metrics_root": str(output_root / "final_metrics"),
        "results_root": str(output_root / "results"),
    })
    out = output_root / "gpt_oss_20b_semantic_metrics_config.json"
    write_json(out, base)
    print(out)


if __name__ == "__main__":
    main()
