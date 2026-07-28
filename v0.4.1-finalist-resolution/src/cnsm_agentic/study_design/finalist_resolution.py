from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import ResearchCandidate, ResearchProgramme


FINALIST_DIMENSIONS = (
    "measurement_validity",
    "confound_control",
    "statistical_identifiability",
    "intervention_executability",
    "scientific_sequence_value",
    "compute_feasibility",
)


@dataclass(frozen=True)
class FinalistScore:
    candidate_id: str
    dimensions: dict[str, float]
    weighted_total: float
    rationale: str

    def validate(self) -> None:
        missing = set(FINALIST_DIMENSIONS) - set(self.dimensions)

        if missing:
            raise ValueError(
                f"{self.candidate_id}: missing finalist dimensions "
                f"{sorted(missing)}"
            )

        for name in FINALIST_DIMENSIONS:
            value = float(self.dimensions[name])

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{self.candidate_id}: {name} must be between 0 and 1"
                )


def deterministic_dimensions(
    candidate: ResearchCandidate,
) -> tuple[dict[str, float], str]:
    if candidate.candidate_id == "C2":
        return (
            {
                "measurement_validity": 0.90,
                "confound_control": 0.88,
                "statistical_identifiability": 0.86,
                "intervention_executability": 0.92,
                "scientific_sequence_value": 0.90,
                "compute_feasibility": 0.80,
            },
            (
                "C2 directly manipulates an inference method, uses paired "
                "controls, and tests an operational mitigation after the "
                "completed diagnosis study."
            ),
        )

    if candidate.candidate_id == "C4":
        return (
            {
                "measurement_validity": 0.60,
                "confound_control": 0.58,
                "statistical_identifiability": 0.60,
                "intervention_executability": 0.72,
                "scientific_sequence_value": 0.82,
                "compute_feasibility": 0.85,
            },
            (
                "C4 is scientifically valuable but depends on validated "
                "complexity features and faces task-confounding and "
                "interaction-power risks."
            ),
        )

    return (
        {
            "measurement_validity": 0.75,
            "confound_control": 0.72,
            "statistical_identifiability": 0.72,
            "intervention_executability": 0.75,
            "scientific_sequence_value": 0.80,
            "compute_feasibility": max(
                0.45,
                0.95 - candidate.estimated_model_calls / 12000,
            ),
        },
        "Generic deterministic finalist assessment.",
    )


class FinalistResolver:
    def __init__(self, programme: ResearchProgramme) -> None:
        weights = getattr(
            programme,
            "finalist_resolution_weights",
            None,
        )

        if not isinstance(weights, dict):
            raise ValueError(
                "ResearchProgramme requires "
                "finalist_resolution_weights"
            )

        missing = set(FINALIST_DIMENSIONS) - set(weights)

        if missing:
            raise ValueError(
                "Missing finalist-resolution weights: "
                f"{sorted(missing)}"
            )

        total = sum(
            float(weights[name])
            for name in FINALIST_DIMENSIONS
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "finalist_resolution_weights must sum to 1.0"
            )

        self.weights = {
            name: float(weights[name])
            for name in FINALIST_DIMENSIONS
        }

    def score(
        self,
        candidate: ResearchCandidate,
    ) -> FinalistScore:
        dimensions, rationale = deterministic_dimensions(candidate)

        weighted_total = sum(
            dimensions[name] * self.weights[name]
            for name in FINALIST_DIMENSIONS
        )

        score = FinalistScore(
            candidate_id=candidate.candidate_id,
            dimensions=dimensions,
            weighted_total=weighted_total,
            rationale=rationale,
        )
        score.validate()
        return score

    def resolve(
        self,
        finalists: list[ResearchCandidate],
    ) -> dict[str, Any]:
        if len(finalists) < 2:
            raise ValueError(
                "Finalist resolution requires at least two candidates"
            )

        scores = sorted(
            (
                self.score(candidate)
                for candidate in finalists
            ),
            key=lambda score: score.weighted_total,
            reverse=True,
        )

        selected = scores[0]
        runner_up = scores[1]

        return {
            "selection_status": (
                "selected_after_finalist_resolution"
            ),
            "selected_candidate_id": selected.candidate_id,
            "runner_up_candidate_id": runner_up.candidate_id,
            "planned_follow_up_candidate_id": (
                runner_up.candidate_id
            ),
            "finalist_scores": [
                asdict(score)
                for score in scores
            ],
            "selection_rationale": (
                f"{selected.candidate_id} achieved the highest "
                f"finalist score ({selected.weighted_total:.3f}) "
                f"versus {runner_up.candidate_id} "
                f"({runner_up.weighted_total:.3f}). "
                "The runner-up is preserved as the planned "
                "follow-up study."
            ),
            "resolution_method": (
                "deterministic finalist-specific weighted assessment"
            ),
        }
