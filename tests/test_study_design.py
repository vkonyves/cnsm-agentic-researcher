from __future__ import annotations

import json
from pathlib import Path

import pytest

from cnsm_agentic.study_design.candidates import generate_seed_candidates
from cnsm_agentic.study_design.models import ResearchProgramme
from cnsm_agentic.study_design.planner import build_study_plan
from cnsm_agentic.study_design.state_machine import ResearchState, ResearchStateMachine
from cnsm_agentic.study_design.tournament import StudyDesignTournament


def programme() -> ResearchProgramme:
    path = Path("configs/research_programmes/llm_netops_invariance.json")
    return ResearchProgramme.from_dict(json.loads(path.read_text(encoding="utf-8")))


def test_tournament_selects_candidate() -> None:
    value = programme()
    candidates = generate_seed_candidates(value)
    result = StudyDesignTournament(value).run(candidates)
    assert result["selected_candidate_id"] in {candidate.candidate_id for candidate in candidates}
    assert len(result["critic_reviews"]) == value.candidate_count


def test_plan_has_falsification_and_claim_freeze() -> None:
    value = programme()
    candidates = generate_seed_candidates(value)
    result = StudyDesignTournament(value).run(candidates)
    selected = next(c for c in candidates if c.candidate_id == result["selected_candidate_id"])
    plan = build_study_plan(selected, result)
    plan.validate()
    assert any(node.node_id == "falsification-suite" for node in plan.experiment_nodes)
    assert plan.experiment_nodes[-1].node_id == "freeze-claims"


def test_state_machine_gates() -> None:
    machine = ResearchStateMachine()
    with pytest.raises(ValueError):
        machine.transition(ResearchState.DISCOVERY_COMPLETE, {"mandate_validated"})
    machine.add_gates({"mandate_validated"})
    machine.transition(ResearchState.DISCOVERY_COMPLETE, {"mandate_validated"})
    assert machine.state is ResearchState.DISCOVERY_COMPLETE


def test_state_machine_rejects_skips() -> None:
    machine = ResearchStateMachine()
    machine.add_gates({"x"})
    with pytest.raises(ValueError):
        machine.transition(ResearchState.CANDIDATES_FORMULATED, {"x"})
