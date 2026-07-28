from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

HypothesisKind = Literal["primary", "secondary", "exploratory", "mitigation"]
ClaimStatus = Literal["proposed", "supported", "refuted", "inconclusive"]
NodeStatus = Literal["pending", "ready", "running", "complete", "failed", "blocked"]


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    kind: HypothesisKind
    primary_outcome: str
    null_statement: str
    falsification_condition: str
    analysis_method: str

    def validate(self) -> None:
        values = (
            self.hypothesis_id,
            self.statement,
            self.primary_outcome,
            self.null_statement,
            self.falsification_condition,
            self.analysis_method,
        )
        if any(not value.strip() for value in values):
            raise ValueError(f"Incomplete hypothesis: {self.hypothesis_id}")


@dataclass
class ResearchCandidate:
    candidate_id: str
    title: str
    research_question: str
    scientific_gap: str
    unit_of_analysis: str
    independent_variables: list[str]
    primary_outcome: str
    hypotheses: list[Hypothesis]
    minimum_required_evidence: list[str]
    expected_contribution: str
    failure_conditions: list[str]
    design_summary: str
    estimated_model_calls: int
    estimated_local_compute_hours: float
    tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.candidate_id.strip() or not self.research_question.strip():
            raise ValueError("candidate_id and research_question are required")
        if not self.hypotheses:
            raise ValueError(f"{self.candidate_id}: hypotheses required")
        for hypothesis in self.hypotheses:
            hypothesis.validate()
        if self.estimated_model_calls < 0 or self.estimated_local_compute_hours < 0:
            raise ValueError("Resource estimates cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticReview:
    candidate_id: str
    fatal_confounds: list[str]
    data_leakage_risk: float
    scope_overclaim_risk: float
    weak_baselines: list[str]
    missing_controls: list[str]
    statistical_risks: list[str]
    feasibility_risks: list[str]
    verdict: Literal["pass", "revise", "veto"]
    rationale: str

    def validate(self) -> None:
        for value in (self.data_leakage_risk, self.scope_overclaim_risk):
            if not 0 <= value <= 1:
                raise ValueError("Risk scores must be in [0,1]")


@dataclass(frozen=True)
class ExperimentNode:
    node_id: str
    node_type: str
    depends_on: tuple[str, ...]
    inputs: dict[str, Any]
    outputs: tuple[str, ...]
    gate_requirements: tuple[str, ...]
    status: NodeStatus = "pending"


@dataclass
class StudyPlan:
    study_id: str
    selected_candidate_id: str
    title: str
    research_question: str
    hypotheses: list[Hypothesis]
    preregistration_fields: dict[str, Any]
    experiment_nodes: list[ExperimentNode]
    rejected_candidates: dict[str, str]
    selection_rationale: str

    def validate(self) -> None:
        node_ids = {node.node_id for node in self.experiment_nodes}
        if len(node_ids) != len(self.experiment_nodes):
            raise ValueError("Duplicate experiment node IDs")
        for node in self.experiment_nodes:
            missing = set(node.depends_on) - node_ids
            if missing:
                raise ValueError(f"{node.node_id}: unknown dependencies {sorted(missing)}")


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_type: str
    supporting_artifacts: list[str]
    supporting_sources: list[str]
    statistical_status: ClaimStatus
    scope: dict[str, Any]
    prohibited_generalizations: list[str]

    def validate_for_writing(self) -> None:
        if self.statistical_status == "proposed":
            raise ValueError(f"Claim {self.claim_id} is not frozen")
        if not self.supporting_artifacts and not self.supporting_sources:
            raise ValueError(f"Claim {self.claim_id} has no evidence")


@dataclass
class ResearchProgramme:
    programme_id: str
    title: str
    mandate: str
    domain: str
    benchmark_families: list[str]
    primary_unit: str
    required_controls: list[str]
    candidate_count: int
    selection_weights: dict[str, float]
    minimum_total_score: float
    critic_veto_thresholds: dict[str, float]
    tie_band: float = 0.01

    finalist_resolution_weights: dict[str, float] = field(
        default_factory=lambda: {
            "measurement_validity": 0.25,
            "confound_control": 0.20,
            "statistical_identifiability": 0.20,
            "intervention_executability": 0.15,
            "scientific_sequence_value": 0.10,
            "compute_feasibility": 0.10,
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResearchProgramme":
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least 2")
        if abs(sum(self.selection_weights.values()) - 1.0) > 1e-9:
            raise ValueError("Selection weights must sum to 1.0")
        if not 0 <= self.minimum_total_score <= 1:
            raise ValueError("minimum_total_score must be in [0,1]")
        if not 0.0 <= self.tie_band <= 1.0:
            raise ValueError("tie_band must be between 0 and 1")
        if (
            abs(
                sum(self.finalist_resolution_weights.values())
                - 1.0
            )
            > 1e-9
        ):
            raise ValueError(
                "finalist_resolution_weights must sum to 1.0"
            )
