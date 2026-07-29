from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

class SearchQuery(BaseModel):
    query: str
    rationale: str
    source_preferences: list[str] = Field(default_factory=lambda: ["openalex", "crossref"])


class QueryPlan(BaseModel):
    queries: list[SearchQuery]
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    date_rationale: str


class LiteratureRecord(BaseModel):
    record_id: str
    title: str
    abstract: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    url: str | None = None
    source_api: str
    authors: list[str] = Field(default_factory=list)
    cited_by_count: int | None = None
    retrieved_for_queries: list[str] = Field(default_factory=list)


class EvidenceClaim(BaseModel):
    claim_id: str
    statement: str
    evidence_record_ids: list[str]
    evidence_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class EvidenceSynthesis(BaseModel):
    established_findings: list[EvidenceClaim]
    unresolved_questions: list[EvidenceClaim]
    candidate_gaps: list[EvidenceClaim]
    excluded_claims: list[str] = Field(default_factory=list)


class AutonomousCandidate(BaseModel):
    candidate_id: str
    title: str
    research_question: str
    hypotheses: list[str]
    proposed_design: str
    expected_data: str
    primary_outcome: str
    analysis_outline: str
    novelty_evidence_ids: list[str]
    feasibility_evidence_ids: list[str]
    risks: list[str]
    estimated_model_calls: int

    @field_validator(
        "title",
        "research_question",
        "proposed_design",
        "expected_data",
        "primary_outcome",
        "analysis_outline",
    )
    @classmethod
    def reject_invalid_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Field must not be empty")

        if cleaned.upper() in {
            "INVALID",
            "UNKNOWN",
            "TBD",
            "TODO",
            "PLACEHOLDER",
        }:
            raise ValueError(
                f"Invalid placeholder value: {cleaned}"
            )

        return cleaned

    @field_validator("hypotheses")
    @classmethod
    def validate_hypotheses(
        cls,
        values: list[str],
    ) -> list[str]:
        if not values:
            raise ValueError(
                "At least one hypothesis is required"
            )

        for hypothesis in values:
            stripped = hypothesis.strip()

            if not stripped:
                raise ValueError(
                    "Hypotheses must not be empty"
                )

            if (
                '"proposed_design"' in stripped
                or '"expected_data"' in stripped
                or '"primary_outcome"' in stripped
                or '"analysis_outline"' in stripped
            ):
                raise ValueError(
                    "Hypothesis contains embedded candidate fields"
                )

        return values

    @model_validator(mode="after")
    def validate_candidate_completeness(
        self,
    ) -> "AutonomousCandidate":
        if self.estimated_model_calls <= 0:
            raise ValueError(
                "estimated_model_calls must be positive"
            )

        if not self.novelty_evidence_ids:
            raise ValueError(
                "At least one novelty evidence ID is required"
            )

        if not self.feasibility_evidence_ids:
            raise ValueError(
                "At least one feasibility evidence ID is required"
            )

        return self


class CandidateSet(BaseModel):
    candidates: list[AutonomousCandidate]


class CandidateReview(BaseModel):
    candidate_id: str
    novelty: float = Field(ge=0.0, le=1.0)
    falsifiability: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    causal_interpretability: float = Field(ge=0.0, le=1.0)
    reproducibility: float = Field(ge=0.0, le=1.0)
    compute_feasibility: float = Field(ge=0.0, le=1.0)
    venue_relevance: float = Field(ge=0.0, le=1.0)
    major_concerns: list[str]
    required_design_repairs: list[str]
    verdict: str


class ReviewSet(BaseModel):
    reviews: list[CandidateReview]


class SelectionDecision(BaseModel):
    selected_candidate_id: str
    selection_rationale: str
    runner_up_candidate_id: str | None = None
    unresolved_uncertainties: list[str]
    evidence_record_ids: list[str]
    required_repairs_before_preregistration: list[str]
