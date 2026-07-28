from __future__ import annotations

from dataclasses import asdict

from .critic import deterministic_critic
from .models import ResearchCandidate, ResearchProgramme
from .scoring import CandidateScore


def _scores(candidate: ResearchCandidate) -> dict[str, float]:
    mitigation = 0.08 if "mitigation" in candidate.tags else 0.0
    mechanism = 0.05 if "mechanism" in candidate.tags else 0.0

    return {
        "novelty": min(0.72 + mitigation + mechanism, 0.95),
        "falsifiability": 0.90,
        "causal_interpretability": (
            0.88 if "paired" in candidate.tags else 0.72
        ),
        "reproducibility": 0.90,
        "compute_feasibility": max(
            0.45,
            0.95 - candidate.estimated_model_calls / 12000,
        ),
        "data_accessibility": 0.92,
        "statistical_power": 0.84,
        "venue_relevance": 0.90,
    }


class StudyDesignTournament:
    def __init__(self, programme: ResearchProgramme) -> None:
        self.programme = programme

    def run(
        self,
        candidates: list[ResearchCandidate],
    ) -> dict[str, object]:
        if not candidates:
            raise ValueError("At least one research candidate is required.")

        reviews = [
            deterministic_critic(candidate, self.programme)
            for candidate in candidates
        ]

        for review in reviews:
            review.validate()

        scores = [
            CandidateScore.calculate(
                candidate.candidate_id,
                _scores(candidate),
                self.programme.selection_weights,
            )
            for candidate, review in zip(candidates, reviews)
            if review.verdict != "veto"
        ]

        eligible = [
            score
            for score in scores
            if score.weighted_total
            >= self.programme.minimum_total_score
        ]

        if not eligible:
            raise RuntimeError(
                "No candidate passed the tournament threshold."
            )

        sorted_scores = sorted(
            eligible,
            key=lambda score: score.weighted_total,
            reverse=True,
        )

        winner = sorted_scores[0]
        tie_band = self.programme.tie_band

        finalist_scores = [
            score
            for score in sorted_scores
            if (
                winner.weighted_total - score.weighted_total
                <= tie_band
            )
        ]

        finalist_ids = [
            score.candidate_id
            for score in finalist_scores
        ]

        is_tie = len(finalist_scores) > 1

        rejected: dict[str, str] = {}

        review_by_candidate = {
            review.candidate_id: review
            for review in reviews
        }

        score_by_candidate = {
            score.candidate_id: score
            for score in scores
        }

        for candidate in candidates:
            candidate_id = candidate.candidate_id
            review = review_by_candidate[candidate_id]

            if review.verdict == "veto":
                rejected[candidate_id] = (
                    "Critic veto: "
                    f"{review.rationale}"
                )
                continue

            score = score_by_candidate.get(candidate_id)

            if score is None:
                rejected[candidate_id] = (
                    "Candidate was not scored."
                )
                continue

            if score.weighted_total < (
                self.programme.minimum_total_score
            ):
                rejected[candidate_id] = (
                    "Candidate did not reach the minimum "
                    "tournament threshold: "
                    f"{score.weighted_total:.3f} < "
                    f"{self.programme.minimum_total_score:.3f}."
                )
                continue

            if candidate_id in finalist_ids:
                continue

            rejected[candidate_id] = (
                f"Eligible but scored "
                f"{score.weighted_total:.3f}, outside the "
                f"tie band of {tie_band:.3f} below the "
                f"leading score of "
                f"{winner.weighted_total:.3f}."
            )

        common_payload = {
            "finalist_candidate_ids": finalist_ids,
            "scores": [
                asdict(score)
                for score in sorted_scores
            ],
            "critic_reviews": [
                asdict(review)
                for review in reviews
            ],
            "rejected_candidates": rejected,
            "tie_band": tie_band,
            "leading_score": winner.weighted_total,
        }

        if is_tie:
            return {
                "selected_candidate_id": None,
                "selection_status": (
                    "tie_resolution_required"
                ),
                **common_payload,
                "selection_rationale": (
                    "No final candidate was selected because "
                    "multiple candidates fell within the "
                    f"tie band of {tie_band:.3f}. "
                    f"Finalists: {', '.join(finalist_ids)}. "
                    "A finalist-resolution stage is required."
                ),
            }

        return {
            "selected_candidate_id": winner.candidate_id,
            "selection_status": "selected",
            **common_payload,
            "selection_rationale": (
                f"{winner.candidate_id} achieved the highest "
                "eligible weighted score "
                f"({winner.weighted_total:.3f}) and no other "
                "candidate fell within the configured tie band "
                f"of {tie_band:.3f}."
            ),
        }