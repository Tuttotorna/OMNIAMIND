from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StepProxyResult:
    step_index: int
    emitted_token: str
    split: float
    pressure_from_previous: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioProxyResult:
    scenario_id: str
    split_mean: float
    split_max: float
    pressure_mean: float
    pressure_max: float
    step_results: List[StepProxyResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "split_mean": self.split_mean,
            "split_max": self.split_max,
            "pressure_mean": self.pressure_mean,
            "pressure_max": self.pressure_max,
            "step_results": [step.to_dict() for step in self.step_results],
        }


@dataclass
class ProxyRunResult:
    version: str
    input_file: str
    results: List[ScenarioProxyResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "input_file": self.input_file,
            "results": [result.to_dict() for result in self.results],
        }