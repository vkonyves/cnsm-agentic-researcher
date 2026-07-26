from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


ExperimentFamily = Literal["llm_benchmark", "tabular_ml"]

VerificationStatus = Literal[
    "verified",
    "partially_verified",
    "not_verified",
    "contradicted",
]


def validate_http_url(value: str | None) -> str | None:
    """Validate URLs without adding unsupported JSON Schema formats."""
    if value is None:
        return None

    parsed = urlparse(value)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https")

    if not parsed.netloc:
        raise ValueError("URL must include a hostname")

    return value


class EvidenceSource(BaseModel):
    title: str
    url: str

    source_type: Literal[
        "primary_paper",
        "official_repository",
        "official_dataset_page",
        "official_documentation",
        "secondary_source",
        "other",
    ]

    supports: list[str]
    notes: str

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        validated = validate_http_url(value)
        assert validated is not None
        return validated


class VerifiedResource(BaseModel):
    name: str
    status: VerificationStatus

    paper_identifier: str | None = None
    paper_title: str | None = None

    repository_url: str | None = None

    dataset_urls: list[str]
    dataset_location_notes: str

    licence: str | None = None
    task_types: list[str]

    data_downloadable: bool | None = None
    gold_answers_available: bool | None = None
    evaluation_code_available: bool | None = None

    cpu_feasibility: str

    verified_claims: list[str]
    unresolved_questions: list[str]
    evidence: list[EvidenceSource]

    @field_validator("repository_url")
    @classmethod
    def check_optional_url(cls, value: str | None) -> str | None:
        return validate_http_url(value)

    @field_validator("dataset_urls")
    @classmethod
    def check_dataset_urls(cls, values: list[str]) -> list[str]:
        validated_urls: list[str] = []

        for value in values:
            validated = validate_http_url(value)
            if validated is None:
                raise ValueError("Dataset URL cannot be null")
            validated_urls.append(validated)

        return validated_urls


class DiscoveryReport(BaseModel):
    scope_summary: str
    searched_resources: list[str]
    resources: list[VerifiedResource]

    rejected_or_unresolved_leads: list[str]
    discovery_limitations: list[str]
    next_verification_actions: list[str]


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


class ResearchPlan(BaseModel):
    experiment_family: ExperimentFamily
    scope_summary: str

    candidates: list[CandidateExperiment] = Field(
        min_length=2,
        max_length=3,
    )

    provisional_recommendation: str
    recommendation_reason: str
    recommendation_dependencies: list[str]
    next_machine_actions: list[str]


class CandidateCritique(BaseModel):
    candidate_title: str

    strengths: list[str]
    weaknesses: list[str]

    unsupported_or_weak_claims: list[str]
    feasibility_concerns: list[str]
    novelty_concerns: list[str]
    required_repairs: list[str]

    score_soundness: int = Field(ge=1, le=5)
    score_feasibility: int = Field(ge=1, le=5)
    score_evidence_grounding: int = Field(ge=1, le=5)


class CriticReport(BaseModel):
    critiques: list[CandidateCritique]

    recommended_candidate: str | None
    selection_reason: str
    blocking_issues: list[str]
    proceed_to_dataset_acquisition: bool


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
