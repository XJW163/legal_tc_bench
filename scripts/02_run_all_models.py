from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import advertised_models, get_model, load_config, model_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["glossary", "placeholder"], default=["glossary", "placeholder"])
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()

    config, root = load_config(args.config)
    selected = args.models or model_keys(config)
    script = Path(__file__).with_name("01_run_model.py")
    failures = []
    for key in selected:
        model = get_model(config, key)
        if not args.skip_server_check:
            try:
                available = advertised_models(str(model["base_url"]))
                if str(model["model"]) not in available:
                    message = f"{key}: endpoint has {available}, expected {model['model']}"
                    if args.skip_unavailable:
                        print("SKIP", message)
                        continue
                    raise RuntimeError(message)
            except Exception as exc:
                if args.skip_unavailable:
                    print("SKIP", key, exc)
                    continue
                raise
        cmd = [sys.executable, str(script), key, "--config", args.config, "--methods", *args.methods]
        if args.skip_server_check:
            cmd.append("--skip-server-check")
        if args.max_documents:
            cmd.extend(["--max-documents", str(args.max_documents)])
        completed = subprocess.run(cmd, cwd=root)
        if completed.returncode != 0:
            failures.append(key)
            if not args.skip_unavailable:
                break
    if failures:
        raise SystemExit("failed model(s): " + ", ".join(failures))


if __name__ == "__main__":
    main()
