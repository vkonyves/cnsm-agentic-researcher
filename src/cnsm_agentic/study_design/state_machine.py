from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ResearchState(str, Enum):
    MANDATE_RECEIVED = "MANDATE_RECEIVED"
    DISCOVERY_COMPLETE = "DISCOVERY_COMPLETE"
    CANDIDATES_FORMULATED = "CANDIDATES_FORMULATED"
    DESIGN_SELECTED = "DESIGN_SELECTED"
    PREREGISTRATION_FROZEN = "PREREGISTRATION_FROZEN"
    DATA_FROZEN = "DATA_FROZEN"
    EXPERIMENTS_RUNNING = "EXPERIMENTS_RUNNING"
    EXPERIMENTS_COMPLETE = "EXPERIMENTS_COMPLETE"
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    FALSIFICATION_PASSED = "FALSIFICATION_PASSED"
    CLAIMS_FROZEN = "CLAIMS_FROZEN"
    DRAFT_COMPLETE = "DRAFT_COMPLETE"
    REVIEW_COMPLETE = "REVIEW_COMPLETE"
    REVISION_COMPLETE = "REVISION_COMPLETE"
    RELEASE_READY = "RELEASE_READY"


_ORDER = list(ResearchState)


@dataclass
class ResearchStateMachine:
    state: ResearchState = ResearchState.MANDATE_RECEIVED
    satisfied_gates: set[str] = field(default_factory=set)

    def add_gates(self, gates: Iterable[str]) -> None:
        self.satisfied_gates.update(gates)

    def transition(self, target: ResearchState, required_gates: Iterable[str]) -> None:
        if _ORDER.index(target) != _ORDER.index(self.state) + 1:
            raise ValueError(f"Invalid transition {self.state.value} -> {target.value}")
        missing = set(required_gates) - self.satisfied_gates
        if missing:
            raise ValueError(f"Missing gates: {sorted(missing)}")
        self.state = target

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state.value, "satisfied_gates": sorted(self.satisfied_gates)}
