from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Candidate:
    token: str
    prob: float


@dataclass
class Step:
    token: str
    candidates: List[Candidate]


@dataclass
class ScenarioResult:
    scenario_id: str
    split_mean: float
    split_max: float
    pressure_mean: float
    pressure_max: float
    step_results: List[Dict[str, Any]]


def normalize_probs(candidates: List[Candidate]) -> List[Candidate]:
    total = sum(max(c.prob, 0.0) for c in candidates)
    if total <= 0:
        raise ValueError("Candidate probabilities must sum to a positive value.")
    return [Candidate(token=c.token, prob=max(c.prob, 0.0) / total) for c in candidates]


def split_proxy(candidates: List[Candidate]) -> float:
    """
    Structural split proxy in [0, 1].

    Current version:
    S = 1 - p_max

    Interpretation:
    - near 0: one candidate dominates
    - near 1: probability mass is fragmented
    """
    if not candidates:
        raise ValueError("split_proxy requires at least one candidate.")
    candidates = normalize_probs(candidates)
    p_max = max(c.prob for c in candidates)
    return 1.0 - p_max


def candidate_prob_map(candidates: List[Candidate]) -> Dict[str, float]:
    candidates = normalize_probs(candidates)
    return {c.token: c.prob for c in candidates}


def bifurcation_pressure(
    prev_candidates: List[Candidate],
    curr_candidates: List[Candidate],
) -> float:
    """
    Structural bifurcation pressure proxy in [0, 1] approximately.

    Current version combines:
    1. distribution shift across consecutive steps
    2. top-rank swap signal

    Distribution shift:
    TV = 0.5 * sum_i |p_t(i) - p_t-1(i)|

    Rank swap:
    1 if top-1 token changes, else 0

    Composite:
    B = 0.7 * TV + 0.3 * rank_swap
    """
    prev_map = candidate_prob_map(prev_candidates)
    curr_map = candidate_prob_map(curr_candidates)

    vocab = set(prev_map) | set(curr_map)
    tv = 0.5 * sum(abs(prev_map.get(tok, 0.0) - curr_map.get(tok, 0.0)) for tok in vocab)

    prev_top = max(prev_map.items(), key=lambda x: x[1])[0]
    curr_top = max(curr_map.items(), key=lambda x: x[1])[0]
    rank_swap = 1.0 if prev_top != curr_top else 0.0

    return 0.7 * tv + 0.3 * rank_swap


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


def evaluate_scenario(scenario_id: str, steps: List[Step]) -> ScenarioResult:
    step_results: List[Dict[str, Any]] = []

    split_values: List[float] = []
    pressure_values: List[float] = []

    prev_step: Optional[Step] = None

    for idx, step in enumerate(steps):
        split_value = split_proxy(step.candidates)
        split_values.append(split_value)

        pressure_value: Optional[float] = None
        if prev_step is not None:
            pressure_value = bifurcation_pressure(prev_step.candidates, step.candidates)
            pressure_values.append(pressure_value)

        step_results.append(
            {
                "step_index": idx,
                "emitted_token": step.token,
                "split": round(split_value, 6),
                "pressure_from_previous": None if pressure_value is None else round(pressure_value, 6),
            }
        )

        prev_step = step

    return ScenarioResult(
        scenario_id=scenario_id,
        split_mean=sum(split_values) / len(split_values) if split_values else 0.0,
        split_max=max(split_values) if split_values else 0.0,
        pressure_mean=sum(pressure_values) / len(pressure_values) if pressure_values else 0.0,
        pressure_max=max(pressure_values) if pressure_values else 0.0,
        step_results=step_results,
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, results: List[ScenarioResult]) -> None:
    payload = {
        "version": "v0",
        "results": [
            {
                "scenario_id": r.scenario_id,
                "split_mean": round(r.split_mean, 6),
                "split_max": round(r.split_max, 6),
                "pressure_mean": round(r.pressure_mean, 6),
                "pressure_max": round(r.pressure_max, 6),
                "step_results": r.step_results,
            }
            for r in results
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    input_path = Path("data/synthetic_proxy_traces_v0.json")
    output_path = Path("data/synthetic_proxy_results_v0.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_json(input_path)
    scenarios = parse_scenarios(data)

    results: List[ScenarioResult] = []
    for scenario_id, steps in scenarios:
        result = evaluate_scenario(scenario_id, steps)
        results.append(result)

    save_results(output_path, results)

    print(f"Wrote results to: {output_path}")
    for r in results:
        print(
            f"{r.scenario_id}: "
            f"split_mean={r.split_mean:.4f}, "
            f"split_max={r.split_max:.4f}, "
            f"pressure_mean={r.pressure_mean:.4f}, "
            f"pressure_max={r.pressure_max:.4f}"
        )


if __name__ == "__main__":
    main()