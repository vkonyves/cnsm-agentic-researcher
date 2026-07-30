from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cnsm_agentic.study_design.candidates import (
    generate_seed_candidates,
)
from cnsm_agentic.study_design.finalist_resolution import (
    FinalistResolver,
)
from cnsm_agentic.study_design.models import (
    ResearchCandidate,
    ResearchProgramme,
)
from cnsm_agentic.study_design.planner import (
    build_study_plan,
)
from cnsm_agentic.study_design.state_machine import (
    ResearchState,
    ResearchStateMachine,
)
from cnsm_agentic.study_design.tournament import (
    StudyDesignTournament,
)


def programme() -> ResearchProgramme:
    path = Path(
        "configs/research_programmes/"
        "llm_netops_invariance.json"
    )

    return ResearchProgramme.from_dict(
        json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    )


def resolve_tournament_selection(
    *,
    value: ResearchProgramme,
    candidates: list[ResearchCandidate],
    tournament_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a result containing a selected candidate.

    A tournament may either select directly or deliberately
    stop with tie_resolution_required. In the latter case,
    run the deterministic finalist-resolution stage and merge
    its result with the tournament evidence.
    """
    selected_candidate_id = (
        tournament_result.get(
            "selected_candidate_id"
        )
    )

    if selected_candidate_id:
        return tournament_result

    selection_status = (
        tournament_result.get(
            "selection_status"
        )
    )

    if (
        selection_status
        != "tie_resolution_required"
    ):
        raise AssertionError(
            "Tournament returned no selected candidate "
            "without requesting finalist resolution. "
            f"Status: {selection_status!r}"
        )

    finalist_ids = (
        tournament_result.get(
            "finalist_candidate_ids"
        )
        or []
    )

    if len(finalist_ids) < 2:
        raise AssertionError(
            "Tie resolution requires at least "
            "two finalist candidate IDs."
        )

    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    unknown_ids = sorted(
        set(finalist_ids)
        - set(candidates_by_id)
    )

    if unknown_ids:
        raise AssertionError(
            "Tournament returned unknown finalist IDs: "
            f"{unknown_ids}"
        )

    finalists = [
        candidates_by_id[candidate_id]
        for candidate_id in finalist_ids
    ]

    resolution = FinalistResolver(
        value
    ).resolve(
        finalists
    )

    # Preserve tournament scores, reviews, rejected candidates,
    # and tie information while adding the final resolution.
    return {
        **tournament_result,
        **resolution,
        "tournament_selection_status": (
            tournament_result[
                "selection_status"
            ]
        ),
        "tournament_finalist_candidate_ids": (
            finalist_ids
        ),
    }


def test_tournament_selects_or_resolves_candidate() -> None:
    value = programme()
    candidates = generate_seed_candidates(
        value
    )

    tournament_result = (
        StudyDesignTournament(
            value
        ).run(
            candidates
        )
    )

    candidate_ids = {
        candidate.candidate_id
        for candidate in candidates
    }

    assert len(
        tournament_result[
            "critic_reviews"
        ]
    ) == value.candidate_count

    if (
        tournament_result[
            "selection_status"
        ]
        == "tie_resolution_required"
    ):
        assert (
            tournament_result[
                "selected_candidate_id"
            ]
            is None
        )

        finalist_ids = set(
            tournament_result[
                "finalist_candidate_ids"
            ]
        )

        assert len(finalist_ids) >= 2
        assert finalist_ids <= candidate_ids

    resolved_result = (
        resolve_tournament_selection(
            value=value,
            candidates=candidates,
            tournament_result=(
                tournament_result
            ),
        )
    )

    assert (
        resolved_result[
            "selected_candidate_id"
        ]
        in candidate_ids
    )

    assert resolved_result[
        "selection_status"
    ] in {
        "selected",
        "selected_after_finalist_resolution",
    }


def test_plan_has_falsification_and_claim_freeze() -> None:
    value = programme()
    candidates = generate_seed_candidates(
        value
    )

    tournament_result = (
        StudyDesignTournament(
            value
        ).run(
            candidates
        )
    )

    resolved_result = (
        resolve_tournament_selection(
            value=value,
            candidates=candidates,
            tournament_result=(
                tournament_result
            ),
        )
    )

    selected_candidate_id = (
        resolved_result[
            "selected_candidate_id"
        ]
    )

    selected = next(
        candidate
        for candidate in candidates
        if (
            candidate.candidate_id
            == selected_candidate_id
        )
    )

    plan = build_study_plan(
        selected,
        resolved_result,
    )

    plan.validate()

    assert any(
        node.node_id
        == "falsification-suite"
        for node in plan.experiment_nodes
    )

    assert (
        plan.experiment_nodes[-1].node_id
        == "freeze-claims"
    )


def test_state_machine_gates() -> None:
    machine = ResearchStateMachine()

    with pytest.raises(
        ValueError
    ):
        machine.transition(
            ResearchState.DISCOVERY_COMPLETE,
            {
                "mandate_validated"
            },
        )

    machine.add_gates(
        {
            "mandate_validated"
        }
    )

    machine.transition(
        ResearchState.DISCOVERY_COMPLETE,
        {
            "mandate_validated"
        },
    )

    assert (
        machine.state
        is ResearchState.DISCOVERY_COMPLETE
    )


def test_state_machine_rejects_skips() -> None:
    machine = ResearchStateMachine()

    machine.add_gates(
        {
            "x"
        }
    )

    with pytest.raises(
        ValueError
    ):
        machine.transition(
            ResearchState.CANDIDATES_FORMULATED,
            {
                "x"
            },
        )
