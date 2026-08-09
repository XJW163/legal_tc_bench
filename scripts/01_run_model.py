from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    atomic_json,
    controlled_experiment_name,
    get_model,
    load_config,
    model_keys,
    p,
    require_model_available,
)


def install_request_adapter(model_config: dict[str, Any]) -> None:
    """Inject model-specific request options without replacing project source files."""
    import legal_tc_bench.controlled_translation as ct

    extra_body = model_config.get("request_extra_body")
    request_parameters = dict(model_config.get("request_parameters") or {})

    def request_controlled_completion(
        *,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        kwargs.update(request_parameters)
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise ct.ControlledTranslationTruncatedError("controlled translation truncated by max_tokens")
        output = ct._clean_translation(choice.message.content)
        if not output:
            raise ValueError("model returned empty translation")
        return output

    ct._request_controlled_completion = request_controlled_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--methods", nargs="+", choices=["glossary", "placeholder"], default=["glossary", "placeholder"])
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--document-ids", help="comma-separated exact document IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    config, root = load_config(args.config)
    if args.model_key not in model_keys(config):
        raise SystemExit(f"unknown model_key {args.model_key!r}")
    model_config = get_model(config, args.model_key)
    if not args.skip_server_check:
        available = require_model_available(model_config)
        print("advertised models:", available)

    install_request_adapter(model_config)
    from legal_tc_bench.controlled_translation import controlled_batch_translate

    paths = config["paths"]
    defaults = config["translation_defaults"]
    annotated = p(root, paths["annotated"])
    term_memory = p(root, paths["term_memory"])
    occurrence_validation = p(root, paths["occurrence_validation"])
    output_root = p(root, paths["controlled_root"])
    document_ids = {x.strip() for x in args.document_ids.split(",") if x.strip()} if args.document_ids else None

    summaries = []
    for method in args.methods:
        experiment = controlled_experiment_name(args.model_key, method)
        result = controlled_batch_translate(
            annotated,
            term_memory,
            output_root,
            experiment_name=experiment,
            model=str(model_config["model"]),
            base_url=str(model_config["base_url"]),
            constraint_mode=method,
            temperature=float(defaults["temperature"]),
            max_tokens=int(defaults["max_tokens"]),
            context_paragraphs=int(defaults["context_paragraphs"]),
            max_context_chars=int(defaults["max_context_chars"]),
            max_retries=int(defaults["max_retries"]),
            timeout_seconds=float(defaults["timeout_seconds"]),
            workers=int(defaults["workers"]),
            overwrite=args.overwrite,
            continue_on_error=bool(defaults.get("continue_on_error", True)),
            max_documents=args.max_documents,
            document_ids=document_ids,
            occurrence_validation=occurrence_validation,
            placeholder_repair_policy=str(defaults["placeholder_repair_policy"]),
            placeholder_quality_gate=True,
            placeholder_semantic_verifier=(
                str(defaults["placeholder_semantic_verifier"]) if method == "placeholder" else "off"
            ),
            placeholder_repeat_rejection_limit=int(defaults["placeholder_repeat_rejection_limit"]),
            semantic_verifier_retries=int(defaults["semantic_verifier_retries"]),
        )
        manifest_path = Path(result["manifest"])
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["request_adapter"] = {
                "request_extra_body": model_config.get("request_extra_body"),
                "request_parameters": model_config.get("request_parameters") or {},
            }
            tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(manifest_path)
        except (OSError, json.JSONDecodeError):
            pass
        summaries.append({
            "model_key": args.model_key,
            "method": method,
            "request_adapter": {
                "request_extra_body": model_config.get("request_extra_body"),
                "request_parameters": model_config.get("request_parameters") or {},
            },
            **result,
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))

    output = p(root, paths["results_root"]) / "run_summaries" / f"{args.model_key}.json"
    atomic_json(output, {"model_key": args.model_key, "runs": summaries})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
