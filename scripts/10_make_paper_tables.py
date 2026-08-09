from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import load_config, load_json, p, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_experiment.json")
    args = parser.parse_args()

    config, root = load_config(args.config)
    results_root = p(root, config["paths"]["results_root"])
    comparison = load_json(results_root / "model_comparison.json", {}) or {}
    reliability = read_csv(results_root / "controlled_reliability.csv")
    reliability_by = {(r.get("model_key"), r.get("method")): r for r in reliability}

    main_rows = []
    for row in comparison.get("models", []):
        exp = str(row.get("model", ""))
        if "__" in exp:
            model_key, method = exp.split("__", 1)
        else:
            model_key, method = exp, "unknown"
        main_rows.append({
            "Model": model_key,
            "Method": method,
            "DTR_macro": row.get("DTR_macro"),
            "DTR_micro": row.get("DTR_micro"),
            "Entropy_macro": row.get("Entropy_macro"),
            "SLCR": row.get("SLCR"),
            "LIR": row.get("LIR"),
            "Omission_rate": row.get("Omission_rate"),
            "Explicit_coverage": row.get("Explicit_coverage"),
            "Critical_risk_rate": row.get("Critical_risk_rate"),
            "WLCS": row.get("WLCS"),
        })
    write_csv(results_root / "table1_main_results.csv", main_rows)

    reliability_rows = []
    for model in config["models"]:
        key = model["key"]
        row = reliability_by.get((key, "placeholder"), {})
        reliability_rows.append({
            "Model": key,
            "First_pass_rate": row.get("first_pass_rate"),
            "Retry_recovery_rate": row.get("retry_recovery_rate"),
            "Placeholder_path_success_rate": row.get("placeholder_path_success_rate"),
            "Fallback_rate": row.get("fallback_rate"),
            "Final_compliance_rate": row.get("final_compliance_rate"),
            "Unresolved_rate": row.get("unresolved_rate"),
            "Verifier_calls": row.get("verifier_calls"),
            "Seconds_per_document": row.get("seconds_per_document"),
        })
    write_csv(results_root / "table2_closed_loop_reliability.csv", reliability_rows)

    md_rows = [[
        r["Model"], r["Method"], fmt(r["DTR_macro"]), fmt(r["SLCR"]), fmt(r["LIR"]), fmt(r["Omission_rate"]), fmt(r["WLCS"])
    ] for r in main_rows]
    rel_md_rows = [[
        r["Model"], fmt(r["First_pass_rate"]), fmt(r["Placeholder_path_success_rate"]), fmt(r["Fallback_rate"]), fmt(r["Final_compliance_rate"]), fmt(r["Unresolved_rate"])
    ] for r in reliability_rows]
    markdown = "# Final experiment tables\n\n## Main results\n\n" + markdown_table(
        ["Model", "Method", "DTR-macro", "SLCR", "LIR", "Omission", "WLCS"], md_rows
    ) + "\n\n## Closed-loop reliability\n\n" + markdown_table(
        ["Model", "First pass", "Placeholder success", "Fallback", "Final compliance", "Unresolved"], rel_md_rows
    ) + "\n"
    (results_root / "paper_tables.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {results_root / 'paper_tables.md'}")


if __name__ == "__main__":
    main()
