from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from result_schema import ProxyRunResult, ScenarioProxyResult, StepProxyResult


@dataclass
class Candidate:
    token: str
    prob: float


@dataclass
class Step:
    token: str
    candidates: List[Candidate]


def normalize_probs(candidates: List[Candidate]) -> List[Candidate]:
    total = sum(max(c.prob, 0.0) for c in candidates)
    if total <= 0:
        raise ValueError("Candidate probabilities must sum to a positive value.")
    return [Candidate(token=c.token, prob=max(c.prob, 0.0) / total) for c in candidates]


def split_proxy(candidates: List[Candidate]) -> float:
    """
    Split proxy v0:
        S_t = 1 - max_i P_t(c_i)

    Interpretation:
    - near 0: dominant candidate
    - near 1: fragmented candidate mass
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
    Bifurcation Pressure proxy v0:
        B_t = 0.7 * TV_t + 0.3 * RS_t

    where:
    - TV_t = total variation shift across adjacent steps
    - RS_t = 1 if top-ranked candidate changes, else 0
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


def evaluate_scenario(scenario_id: str, steps: List[Step]) -> ScenarioProxyResult:
    step_results: List[StepProxyResult] = []

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
            StepProxyResult(
                step_index=idx,
                emitted_token=step.token,
                split=round(split_value, 6),
                pressure_from_previous=None if pressure_value is None else round(pressure_value, 6),
            )
        )

        prev_step = step

    return ScenarioProxyResult(
        scenario_id=scenario_id,
        split_mean=round(sum(split_values) / len(split_values), 6) if split_values else 0.0,
        split_max=round(max(split_values), 6) if split_values else 0.0,
        pressure_mean=round(sum(pressure_values) / len(pressure_values), 6) if pressure_values else 0.0,
        pressure_max=round(max(pressure_values), 6) if pressure_values else 0.0,
        step_results=step_results,
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, run_result: ProxyRunResult) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(run_result.to_dict(), f, indent=2)


def main() -> None:
    input_path = Path("data/synthetic_proxy_traces_v0.json")
    output_path = Path("data/synthetic_proxy_results_v0.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_json(input_path)
    scenarios = parse_scenarios(data)

    results: List[ScenarioProxyResult] = []
    for scenario_id, steps in scenarios:
        results.append(evaluate_scenario(scenario_id, steps))

    run_result = ProxyRunResult(
        version="v0",
        input_file=str(input_path),
        results=results,
    )

    save_results(output_path, run_result)

    print(f"Wrote results to: {output_path}")
    for result in results:
        print(
            f"{result.scenario_id}: "
            f"split_mean={result.split_mean:.4f}, "
            f"split_max={result.split_max:.4f}, "
            f"pressure_mean={result.pressure_mean:.4f}, "
            f"pressure_max={result.pressure_max:.4f}"
        )


if __name__ == "__main__":
    main()