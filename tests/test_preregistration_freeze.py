from __future__ import annotations

import hashlib

from cnsm_agentic.study_design.preregistration import (
    build_preregistration,
    build_run_specs,
    canonical_json_bytes,
)


def inputs() -> tuple[dict, dict, dict]:
    return (
        {"programme_id": "llm-netops-decision-invariance"},
        {"title": "Structured reasoning mitigation"},
        {
            "selected_candidate_id": "C2",
            "planned_follow_up_candidate_id": "C4",
        },
    )


def test_preregistration_is_deterministic() -> None:
    programme, study, resolution = inputs()

    first = build_preregistration(
        programme=programme,
        selected_study=study,
        finalist_resolution=resolution,
    )
    second = build_preregistration(
        programme=programme,
        selected_study=study,
        finalist_resolution=resolution,
    )

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["design"]["planned_evaluations"] == 3600


def test_six_run_specs() -> None:
    programme, study, resolution = inputs()

    prereg = build_preregistration(
        programme=programme,
        selected_study=study,
        finalist_resolution=resolution,
    )
    specs = build_run_specs(prereg)

    assert len(specs) == 6
    assert sum(spec.sample_size for spec in specs) == 3600


def test_hash_changes_after_mutation() -> None:
    programme, study, resolution = inputs()

    prereg = build_preregistration(
        programme=programme,
        selected_study=study,
        finalist_resolution=resolution,
    )

    first = hashlib.sha256(
        canonical_json_bytes(prereg)
    ).hexdigest()

    prereg["sampling"]["sample_size"] = 601

    second = hashlib.sha256(
        canonical_json_bytes(prereg)
    ).hexdigest()

    assert first != second
