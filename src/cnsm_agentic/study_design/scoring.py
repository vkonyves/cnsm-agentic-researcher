from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DIMENSIONS = (
    "novelty", "falsifiability", "causal_interpretability", "reproducibility",
    "compute_feasibility", "data_accessibility", "statistical_power", "venue_relevance",
)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    dimensions: dict[str, float]
    weighted_total: float

    @classmethod
    def calculate(cls, candidate_id: str, dimensions: Mapping[str, float], weights: Mapping[str, float]) -> "CandidateScore":
        missing = set(DIMENSIONS) - set(dimensions)
        if missing:
            raise ValueError(f"Missing dimensions: {sorted(missing)}")
        for name in DIMENSIONS:
            if not 0 <= float(dimensions[name]) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        total = sum(float(dimensions[name]) * float(weights[name]) for name in DIMENSIONS)
        return cls(candidate_id, dict(dimensions), total)
