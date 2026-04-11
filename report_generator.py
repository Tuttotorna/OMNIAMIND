from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def scenario_interpretation(result: Dict[str, Any]) -> str:
    split_mean = float(result["split_mean"])
    pressure_mean = float(result["pressure_mean"])
    pressure_max = float(result["pressure_max"])

    if split_mean < 0.1 and pressure_max == 0.0:
        return "Monolithic / highly stable"
    if split_mean >= 0.4 and pressure_max >= 0.3:
        return "Strong bifurcation signal"
    if split_mean >= 0.8 and pressure_max == 0.0:
        return "Flat collapse / fragmented single-step state"
    return "Intermediate / mixed structural regime"


def generate_report(data: Dict[str, Any]) -> str:
    version = data.get("version", "unknown")
    input_file = data.get("input_file", "unknown")
    results: List[Dict[str, Any]] = data.get("results", [])

    lines: List[str] = []
    lines.append("# OMNIAMIND — Synthetic Proxy Report")
    lines.append("")
    lines.append(f"**Version:** {version}")
    lines.append(f"**Input file:** `{input_file}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Scenario | Split Mean | Split Max | Pressure Mean | Pressure Max | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for result in results:
        lines.append(
            f"| `{result['scenario_id']}` | "
            f"{format_float(float(result['split_mean']))} | "
            f"{format_float(float(result['split_max']))} | "
            f"{format_float(float(result['pressure_mean']))} | "
            f"{format_float(float(result['pressure_max']))} | "
            f"{scenario_interpretation(result)} |"
        )

    lines.append("")
    lines.append("## Per-scenario details")
    lines.append("")

    for result in results:
        lines.append(f"### {result['scenario_id']}")
        lines.append("")
        lines.append(f"- **Split mean:** {format_float(float(result['split_mean']))}")
        lines.append(f"- **Split max:** {format_float(float(result['split_max']))}")
        lines.append(f"- **Pressure mean:** {format_float(float(result['pressure_mean']))}")
        lines.append(f"- **Pressure max:** {format_float(float(result['pressure_max']))}")
        lines.append(f"- **Interpretation:** {scenario_interpretation(result)}")
        lines.append("")
        lines.append("| Step | Emitted Token | Split | Pressure From Previous |")
        lines.append("|---:|---|---:|---:|")

        for step in result.get("step_results", []):
            pressure = step["pressure_from_previous"]
            pressure_str = "N/A" if pressure is None else format_float(float(pressure))
            lines.append(
                f"| {step['step_index']} | "
                f"`{step['emitted_token']}` | "
                f"{format_float(float(step['split']))} | "
                f"{pressure_str} |"
            )

        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is based on synthetic traces only.")
    lines.append("- It is a structural sanity check, not empirical validation on real model traces.")
    lines.append("- OMNIAMIND v0 remains strictly non-semantic in this branch.")

    return "\n".join(lines) + "\n"


def main() -> None:
    input_path = Path("data/synthetic_proxy_results_v0.json")
    output_path = Path("reports/synthetic_proxy_report_v0.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    report = generate_report(data)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()