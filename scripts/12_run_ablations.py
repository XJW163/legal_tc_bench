from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json, get_model, load_config, p, require_model_available
from importlib.util import spec_from_file_location, module_from_spec


def load_run_model_module() -> object:
    path = Path(__file__).with_name("01_run_model.py")
    spec = spec_from_file_location("final_run_model", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_document_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("document_ids_file")
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    model = get_model(config, args.model_key)
    if not args.skip_server_check:
        require_model_available(model)
    run_model = load_run_model_module()
    run_model.install_request_adapter(model)
    from legal_tc_bench.controlled_translation import controlled_batch_translate

    paths = config["paths"]
    defaults = config["translation_defaults"]
    document_ids = read_document_ids(Path(args.document_ids_file))
    annotated = p(root, paths["annotated"])
    term_memory = p(root, paths["term_memory"])
    occurrence_validation = p(root, paths["occurrence_validation"])
    output_root = p(root, paths["controlled_root"])

    ablations = [
        {
            "name": "full",
            "semantic": defaults["placeholder_semantic_verifier"],
            "max_retries": defaults["max_retries"],
            "quality_gate": True,
            "repair_policy": defaults["placeholder_repair_policy"],
        },
        {
            "name": "no-semantic-verifier",
            "semantic": "off",
            "max_retries": defaults["max_retries"],
            "quality_gate": True,
            "repair_policy": defaults["placeholder_repair_policy"],
        },
        {
            "name": "no-retry",
            "semantic": "off",
            "max_retries": 1,
            "quality_gate": True,
            "repair_policy": "retry_then_fallback",
        },
        {
            "name": "no-deterministic-quality-gate",
            "semantic": "off",
            "max_retries": defaults["max_retries"],
            "quality_gate": False,
            "repair_policy": defaults["placeholder_repair_policy"],
        },
        {
            "name": "legacy-llm-repair",
            "semantic": "off",
            "max_retries": defaults["max_retries"],
            "quality_gate": True,
            "repair_policy": "llm_repair",
        },
    ]

    results = []
    for ablation in ablations:
        experiment = f"{args.model_key}-ablation-{ablation['name']}"
        result = controlled_batch_translate(
            annotated,
            term_memory,
            output_root,
            experiment_name=experiment,
            model=model["model"],
            base_url=model["base_url"],
            constraint_mode="placeholder",
            temperature=float(defaults["temperature"]),
            max_tokens=int(defaults["max_tokens"]),
            context_paragraphs=int(defaults["context_paragraphs"]),
            max_context_chars=int(defaults["max_context_chars"]),
            max_retries=int(ablation["max_retries"]),
            timeout_seconds=float(defaults["timeout_seconds"]),
            workers=int(defaults["workers"]),
            continue_on_error=True,
            document_ids=document_ids,
            occurrence_validation=occurrence_validation,
            placeholder_repair_policy=ablation["repair_policy"],
            placeholder_quality_gate=bool(ablation["quality_gate"]),
            placeholder_semantic_verifier=ablation["semantic"],
            placeholder_repeat_rejection_limit=int(defaults["placeholder_repeat_rejection_limit"]),
            semantic_verifier_retries=int(defaults["semantic_verifier_retries"]),
        )
        results.append({"ablation": ablation["name"], **result})
        print(json.dumps(result, ensure_ascii=False, indent=2))

    output = p(root, paths["results_root"]) / "ablations" / f"{args.model_key}.json"
    atomic_json(output, {"model_key": args.model_key, "document_ids": sorted(document_ids), "runs": results})


if __name__ == "__main__":
    main()
