from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ExperimentFamily = Literal["llm_benchmark", "tabular_ml"]


class MetricSpec(BaseModel):
    name: str
    definition: str
    required_reference: str
    automatically_computable: bool


class CandidateExperiment(BaseModel):
    title: str
    target_resources: list[str]

    research_question: str
    hypothesis: str

    verified_facts_used: list[str]
    unverified_assumptions: list[str]
    required_verification_actions: list[str]

    dataset_requirements: list[str]
    baselines: list[str]
    metrics: list[MetricSpec]

    cpu_feasibility: str
    estimated_api_call_pattern: str
    key_risks: list[str]


class PilotResearchPlan(BaseModel):
    experiment_family: ExperimentFamily
    scope_summary: str

    candidates: list[CandidateExperiment] = Field(
        min_length=2,
        max_length=3,
    )

    recommended_candidate_title: str
    recommendation_reason: str

    next_machine_actions: list[str]
    human_actions_allowed_before_final_freeze: list[str]
