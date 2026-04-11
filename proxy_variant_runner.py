from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from proxy_variants import (
    Candidate,
    bifurcation_pressure_v0,
    bifurcation_pressure_v01,
    split_v0,
    split_v01,
)


@dataclass
class Step:
    token: str
    candidates: List[Candidate]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_scenarios(data: Dict[str, Any]) -> List[Tuple[str, List[Step]]]:
    scenarios: List[Tuple[str, List[Step]]] = []

    for scenario in data.get("scenarios", []):
        scenario_id = scenario["id"]
        steps_raw = scenario["steps"]
        steps: List[Step] = []

        for step_raw in steps_raw:
            token = step_raw["token"]
            candidates = [
                Candidate(token=c["t"], prob=float(c["p"]))
                for c in step_raw["candidates"]
            ]
            steps.append(Step(token=token, candidates=candidates))

        scenarios.append((scenario_id, steps))

    return scenarios


def evaluate_scenario(scenario_id: str, steps: List[Step]) -> Dict[str, Any]:
    step_results: List[Dict[str, Any]] = []

    split_v0_values: List[float] = []
    split_v01_values: List[float] = []
    pressure_v0_values: List[float] = []
    pressure_v01_values: List[float] = []

    prev_step: Optional[Step] = None

    for idx, step in enumerate(steps):
        s0 = split_v0(step.candidates)
        s01 = split_v01(step.candidates)

        split_v0_values.append(s0)
        split_v01_values.append(s01)

        p0: Optional[float] = None
        p01: Optional[float] = None

        if prev_step is not None:
            p0 = bifurcation_pressure_v0(prev_step.candidates, step.candidates)
            p01 = bifurcation_pressure_v01(prev_step.candidates, step.candidates)

            pressure_v0_values.append(p0)
            pressure_v01_values.append(p01)

        step_results.append(
            {
                "step_index": idx,
                "emitted_token": step.token,
                "split_v0": round(s0, 6),
                "split_v01": round(s01, 6),
                "pressure_v0_from_previous": None if p0 is None else round(p0, 6),
                "pressure_v01_from_previous": None if p01 is None else round(p01, 6),
            }
        )

        prev_step = step

    result = {
        "scenario_id": scenario_id,
        "split_v0_mean": round(sum(split_v0_values) / len(split_v0_values), 6) if split_v0_values else 0.0,
        "split_v0_max": round(max(split_v0_values), 6) if split_v0_values else 0.0,
        "split_v01_mean": round(sum(split_v01_values) / len(split_v01_values), 6) if split_v01_values else 0.0,
        "split_v01_max": round(max(split_v01_values), 6) if split_v01_values else 0.0,
        "pressure_v0_mean": round(sum(pressure_v0_values) / len(pressure_v0_values), 6) if pressure_v0_values else 0.0,
        "pressure_v0_max": round(max(pressure_v0_values), 6) if pressure_v0_values else 0.0,
        "pressure_v01_mean": round(sum(pressure_v01_values) / len(pressure_v01_values), 6) if pressure_v01_values else 0.0,
        "pressure_v01_max": round(max(pressure_v01_values), 6) if pressure_v01_values else 0.0,
        "step_results": step_results,
    }
    return result


def main() -> None:
    input_path = Path("data/synthetic_proxy_traces_v0.json")
    output_path = Path("data/proxy_variant_comparison_v0_1.json")

    data = load_json(input_path)
    scenarios = parse_scenarios(data)

    results = [evaluate_scenario(scenario_id, steps) for scenario_id, steps in scenarios]

    payload = {
        "version": "v0_vs_v0.1",
        "input_file": str(input_path),
        "results": results,
    }

    save_json(output_path, payload)

    print(f"Wrote variant comparison to: {output_path}")
    for result in results:
        print(
            f"{result['scenario_id']}: "
            f"split_v0_mean={result['split_v0_mean']:.4f}, "
            f"split_v01_mean={result['split_v01_mean']:.4f}, "
            f"pressure_v0_max={result['pressure_v0_max']:.4f}, "
            f"pressure_v01_max={result['pressure_v01_max']:.4f}"
        )


if __name__ == "__main__":
    main()