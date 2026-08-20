from pathlib import Path
from types import SimpleNamespace

import pytest

from cnsm_agentic.autonomous_research import pipeline
from cnsm_agentic.autonomous_research.schemas import (
    AutonomousCandidate,
    CandidateSet,
)


def _valid_candidate(candidate_id: str) -> AutonomousCandidate:
    return AutonomousCandidate(
        candidate_id=candidate_id,
        title="Candidate",
        research_question="Does the intervention change the outcome?",
        hypotheses=["H1: The intervention changes the outcome."],
        proposed_design="Paired autonomous experiment.",
        expected_data="Paired binary outcomes.",
        primary_outcome="Paired outcome difference.",
        analysis_outline="Paired statistical comparison.",
        novelty_evidence_ids=["record-1"],
        feasibility_evidence_ids=["record-2"],
        risks=["External validity."],
        estimated_model_calls=20,
    )


@pytest.mark.anyio
async def test_candidate_generation_retry_receives_validation_feedback(
    tmp_path: Path,
    monkeypatch,
):
    payloads = []
    calls = 0

    async def fake_run(agent, payload_json):
        nonlocal calls
        import json

        calls += 1
        payloads.append(json.loads(payload_json))

        if calls == 1:
            raise ValueError(
                "candidates.3.proposed_design: "
                "Field must not be empty"
            )

        return SimpleNamespace(
            final_output=CandidateSet(
                candidates=[
                    _valid_candidate("c1"),
                    _valid_candidate("c2"),
                    _valid_candidate("c3"),
                ]
            )
        )

    async def no_sleep(_):
        return None

    monkeypatch.setattr(
        pipeline.Runner,
        "run",
        fake_run,
    )
    monkeypatch.setattr(
        pipeline.asyncio,
        "sleep",
        no_sleep,
    )

    result = await pipeline._generate_candidate_set_with_repair(
        payload={
            "programme": {"topic": "NetOps"},
            "evidence_synthesis": {},
        },
        attempts_dir=tmp_path / "attempts",
    )

    assert len(result.candidates) == 3
    assert calls == 2

    assert (
        "candidate_generation_repair"
        not in payloads[0]
    )

    repair = payloads[1][
        "candidate_generation_repair"
    ]

    assert repair[
        "previous_output_rejected"
    ] is True

    assert (
        "Field must not be empty"
        in repair[
            "deterministic_validation_error"
        ]
    )

    assert (
        tmp_path
        / "attempts"
        / "candidate_generation_attempt_01_status.json"
    ).exists()

    assert (
        tmp_path
        / "attempts"
        / "candidate_generation_attempt_02_status.json"
    ).exists()
