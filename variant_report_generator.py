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


def compare_interpretation(result: Dict[str, Any]) -> str:
    split_v0 = float(result["split_v0_mean"])
    split_v01 = float(result["split_v01_mean"])
    pressure_v0 = float(result["pressure_v0_max"])
    pressure_v01 = float(result["pressure_v01_max"])

    notes: List[str] = []

    if split_v01 > split_v0:
        notes.append("v0.1 more sensitive to top-2 competition")
    elif split_v01 < split_v0:
        notes.append("v0 more sensitive to broad fragmentation")
    else:
        notes.append("split variants aligned")

    if pressure_v01 > pressure_v0:
        notes.append("v0.1 captures extra dominance-margin instability")
    elif pressure_v01 < pressure_v0:
        notes.append("v0 more conservative on local instability")
    else:
        notes.append("pressure variants aligned")

    return "; ".join(notes)


def generate_report(data: Dict[str, Any]) -> str:
    version = data.get("version", "unknown")
    input_file = data.get("input_file", "unknown")
    results: List[Dict[str, Any]] = data.get("results", [])

    lines: List[str] = []
    lines.append("# OMNIAMIND — Proxy Variant Comparison Report")
    lines.append("")
    lines.append(f"**Version:** {version}")
    lines.append(f"**Input file:** `{input_file}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Scenario | Split v0 Mean | Split v0.1 Mean | Pressure v0 Max | Pressure v0.1 Max | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for result in results:
        lines.append(
            f"| `{result['scenario_id']}` | "
            f"{format_float(float(result['split_v0_mean']))} | "
            f"{format_float(float(result['split_v01_mean']))} | "
            f"{format_float(float(result['pressure_v0_max']))} | "
            f"{format_float(float(result['pressure_v01_max']))} | "
            f"{compare_interpretation(result)} |"
        )

    lines.append("")
    lines.append("## Per-scenario details")
    lines.append("")

    for result in results:
        lines.append(f"### {result['scenario_id']}")
        lines.append("")
        lines.append(f"- **Split v0 mean:** {format_float(float(result['split_v0_mean']))}")
        lines.append(f"- **Split v0.1 mean:** {format_float(float(result['split_v01_mean']))}")
        lines.append(f"- **Pressure v0 mean:** {format_float(float(result['pressure_v0_mean']))}")
        lines.append(f"- **Pressure v0.1 mean:** {format_float(float(result['pressure_v01_mean']))}")
        lines.append(f"- **Pressure v0 max:** {format_float(float(result['pressure_v0_max']))}")
        lines.append(f"- **Pressure v0.1 max:** {format_float(float(result['pressure_v01_max']))}")
        lines.append(f"- **Interpretation:** {compare_interpretation(result)}")
        lines.append("")
        lines.append("| Step | Emitted Token | Split v0 | Split v0.1 | Pressure v0 | Pressure v0.1 |")
        lines.append("|---:|---|---:|---:|---:|---:|")

        for step in result.get("step_results", []):
            p0 = step["pressure_v0_from_previous"]
            p01 = step["pressure_v01_from_previous"]
            p0_str = "N/A" if p0 is None else format_float(float(p0))
            p01_str = "N/A" if p01 is None else format_float(float(p01))

            lines.append(
                f"| {step['step_index']} | "
                f"`{step['emitted_token']}` | "
                f"{format_float(float(step['split_v0']))} | "
                f"{format_float(float(step['split_v01']))} | "
                f"{p0_str} | "
                f"{p01_str} |"
            )

        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This report compares structural proxy variants on synthetic traces only.")
    lines.append("- v0 remains the simplest baseline.")
    lines.append("- v0.1 adds sensitivity to top-2 competition and dominance-gap instability.")
    lines.append("- No semantic layer is used in either variant.")

    return "\n".join(lines) + "\n"


def main() -> None:
    input_path = Path("data/proxy_variant_comparison_v0_1.json")
    output_path = Path("reports/proxy_variant_report_v0_1.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    report = generate_report(data)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote variant report to: {output_path}")


if __name__ == "__main__":
    main()