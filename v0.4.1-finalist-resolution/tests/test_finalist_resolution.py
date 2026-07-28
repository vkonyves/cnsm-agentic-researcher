from __future__ import annotations

import json
from pathlib import Path

from cnsm_agentic.study_design.candidates import (
    generate_seed_candidates,
)
from cnsm_agentic.study_design.finalist_resolution import (
    FinalistResolver,
)
from cnsm_agentic.study_design.models import ResearchProgramme


def load_programme() -> ResearchProgramme:
    path = Path(
        "configs/research_programmes/"
        "llm_netops_invariance.json"
    )

    return ResearchProgramme.from_dict(
        json.loads(
            path.read_text(encoding="utf-8")
        )
    )


def test_c2_wins_and_c4_is_follow_up() -> None:
    programme = load_programme()

    candidates = {
        candidate.candidate_id: candidate
        for candidate in generate_seed_candidates(programme)
    }

    result = FinalistResolver(
        programme
    ).resolve(
        [
            candidates["C4"],
            candidates["C2"],
        ]
    )

    assert result["selected_candidate_id"] == "C2"
    assert result["runner_up_candidate_id"] == "C4"
    assert (
        result["planned_follow_up_candidate_id"]
        == "C4"
    )


def test_finalist_scores_are_sorted() -> None:
    programme = load_programme()

    candidates = {
        candidate.candidate_id: candidate
        for candidate in generate_seed_candidates(programme)
    }

    result = FinalistResolver(
        programme
    ).resolve(
        [
            candidates["C4"],
            candidates["C2"],
        ]
    )

    totals = [
        score["weighted_total"]
        for score in result["finalist_scores"]
    ]

    assert totals == sorted(
        totals,
        reverse=True,
    )
