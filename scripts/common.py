from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(data.get("project_root", ".")).expanduser()
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    return data, project_root


def p(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def get_model(config: dict[str, Any], key: str) -> dict[str, Any]:
    for model in config.get("models", []):
        if model.get("key") == key:
            return model
    known = ", ".join(str(x.get("key")) for x in config.get("models", []))
    raise KeyError(f"unknown model key {key!r}; known: {known}")


def model_keys(config: dict[str, Any]) -> list[str]:
    return [str(x["key"]) for x in config.get("models", [])]


def resolve_baseline_experiment(config: dict[str, Any], root: Path, model: dict[str, Any]) -> str | None:
    baseline_root = p(root, config["paths"]["baseline_root"])
    explicit = model.get("baseline_experiment")
    if explicit and (baseline_root / str(explicit)).is_dir():
        return str(explicit)
    for candidate in model.get("baseline_candidates", []):
        if (baseline_root / str(candidate)).is_dir():
            return str(candidate)
    return str(explicit) if explicit else None


def experiment_id(model_key: str, method: str) -> str:
    return f"{model_key}__{method}"


def controlled_experiment_name(model_key: str, method: str) -> str:
    if method not in {"glossary", "placeholder"}:
        raise ValueError(method)
    return f"{model_key}-{method}-final"


def experiment_registry(config: dict[str, Any], root: Path, *, include_missing: bool = True) -> list[dict[str, Any]]:
    baseline_root = p(root, config["paths"]["baseline_root"])
    controlled_root = p(root, config["paths"]["controlled_root"])
    rows: list[dict[str, Any]] = []
    for model in config.get("models", []):
        key = str(model["key"])
        baseline = resolve_baseline_experiment(config, root, model)
        baseline_path = baseline_root / baseline if baseline else baseline_root / "__missing__"
        item = {
            "experiment_id": experiment_id(key, "baseline"),
            "model_key": key,
            "model": model["model"],
            "method": "baseline",
            "source_experiment": baseline,
            "translation_dir": baseline_path,
        }
        if include_missing or baseline_path.is_dir():
            rows.append(item)
        for method in ("glossary", "placeholder"):
            source_exp = controlled_experiment_name(key, method)
            source_path = controlled_root / source_exp
            item = {
                "experiment_id": experiment_id(key, method),
                "model_key": key,
                "model": model["model"],
                "method": method,
                "source_experiment": source_exp,
                "translation_dir": source_path,
            }
            if include_missing or source_path.is_dir():
                rows.append(item)
    return rows


def run(cmd: Iterable[str | Path], *, cwd: Path, log_path: Path | None = None, check: bool = True) -> int:
    argv = [str(x) for x in cmd]
    print("+", " ".join(argv), flush=True)
    if log_path is None:
        completed = subprocess.run(argv, cwd=cwd)
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, argv)
        return completed.returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n+ " + " ".join(argv) + "\n")
        handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()
    if check and code != 0:
        raise subprocess.CalledProcessError(code, argv)
    return code


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_legal_tc() -> str:
    executable = shutil.which("legal-tc")
    if not executable:
        raise RuntimeError("legal-tc is not on PATH; activate the project virtual environment and run pip install -e .")
    return executable


def advertised_models(base_url: str, timeout: float = 10.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot query {url}: {exc}") from exc
    return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]


def require_model_available(model: dict[str, Any], *, timeout: float = 10.0) -> list[str]:
    available = advertised_models(str(model["base_url"]), timeout=timeout)
    expected = str(model["model"])
    if expected not in available:
        raise RuntimeError(
            f"endpoint {model['base_url']} advertises {available}, not {expected!r}. "
            "Load the requested model or edit config/final_experiment.json."
        )
    return available


def safe_ratio(num: int | float | None, den: int | float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)
