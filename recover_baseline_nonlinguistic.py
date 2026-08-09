#!/usr/bin/env python3
"""Recover baseline failures caused only by document separator paragraphs.

The script reuses each existing result file's exact experiment metadata,
retries only paragraph IDs already recorded as errors, and never sends a
request to the model. Pure underscore/asterisk/whitespace paragraphs are
copied unchanged. Any other failed source aborts the recovery.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import legal_tc_bench.translate as tr


SEPARATOR_ONLY = re.compile(r"[\s_*]+", re.UNICODE)
EXPERIMENT_FIELDS = (
    "name",
    "model",
    "base_url",
    "prompt_profile",
    "temperature",
    "max_tokens",
    "source_chunk_chars",
    "context_paragraphs",
    "max_context_chars",
    "extra_body",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def separator_passthrough(*args, **kwargs):
    source_text = kwargs.get("source_text")
    if (
        isinstance(source_text, str)
        and any(character in source_text for character in "_*")
        and SEPARATOR_ONLY.fullmatch(source_text)
    ):
        return source_text, 0, 0.0, 0
    raise RuntimeError(
        "recovery aborted: an error paragraph is not a pure layout separator: "
        f"{source_text!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotated", type=Path, default=Path("data/annotated")
    )
    parser.add_argument(
        "--translations-root", type=Path, default=Path("data/translations")
    )
    parser.add_argument(
        "--experiment", default="gpt-oss-120b-zero-english-v2"
    )
    args = parser.parse_args()

    experiment_dir = args.translations_root / args.experiment
    output_files = sorted(experiment_dir.glob("*.translation.json"))
    if not output_files:
        raise SystemExit(f"no translation files found in {experiment_dir}")

    tr._translate_source_text = separator_passthrough
    repaired_paragraphs = 0
    reference_experiment: tr.TranslationExperiment | None = None

    for output_path in output_files:
        state = load_json(output_path)
        errors = state.get("errors") or []
        if not errors:
            continue

        for error in errors:
            source = str(error.get("source", ""))
            if not (
                any(character in source for character in "_*")
                and SEPARATOR_ONLY.fullmatch(source)
            ):
                raise SystemExit(
                    "refusing to repair non-separator error: "
                    f"{state.get('document_id')} paragraph "
                    f"{error.get('paragraph_id')}: {source!r}"
                )

        document_id = str(state["document_id"])
        source_path = args.annotated / f"{document_id}.json"
        if not source_path.is_file():
            matches = list(args.annotated.rglob(f"{document_id}.json"))
            if len(matches) != 1:
                raise SystemExit(
                    f"cannot uniquely locate annotated source for {document_id}: "
                    f"{matches}"
                )
            source_path = matches[0]

        experiment_meta = state["experiment"]
        experiment_kwargs = {
            field: experiment_meta[field]
            for field in EXPERIMENT_FIELDS
        }
        experiment = tr.TranslationExperiment(**experiment_kwargs)
        if reference_experiment is None:
            reference_experiment = experiment
        elif experiment != reference_experiment:
            raise SystemExit(
                f"mixed experiment metadata detected in {output_path}"
            )

        result = tr._translate_document(
            dataset=load_json(source_path),
            source_path=source_path,
            output_path=output_path,
            experiment=experiment,
            max_retries=1,
            timeout_seconds=60.0,
            overwrite=False,
            retry_errors_only=True,
            show_progress=False,
        )
        remaining = len(result.get("errors") or [])
        if remaining:
            raise SystemExit(
                f"{output_path} still contains {remaining} errors"
            )
        repaired_paragraphs += len(errors)
        print(
            f"repaired {document_id}: {len(errors)} separator paragraph(s)",
            flush=True,
        )

    if reference_experiment is None:
        print("nothing to repair")
        return

    summary = tr.batch_translate(
        args.annotated,
        args.translations_root,
        experiment_name=reference_experiment.name,
        model=reference_experiment.model,
        base_url=reference_experiment.base_url,
        prompt_profile=reference_experiment.prompt_profile,
        temperature=reference_experiment.temperature,
        max_tokens=reference_experiment.max_tokens,
        source_chunk_chars=reference_experiment.source_chunk_chars,
        context_paragraphs=reference_experiment.context_paragraphs,
        max_context_chars=reference_experiment.max_context_chars,
        extra_body=reference_experiment.extra_body,
        max_retries=1,
        timeout_seconds=60.0,
        overwrite=False,
        continue_on_error=False,
        workers=1,
    )
    print(json.dumps({
        "repaired_paragraphs": repaired_paragraphs,
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

