import pytest

from cnsm_agentic.autonomous_research.controlled_fault_regime import (
    CONTROLLED_FAULT_REGIME_VERSION,
    build_controlled_fault_pair,
    run_deterministic_controlled_fault_pair,
    score_controlled_fault_condition,
)
from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    generate_direct_candidate,
    generate_task,
    render_reference_configuration,
)


def test_pair_uses_identical_injected_candidate_for_both_conditions() -> None:
    task = generate_task(8)
    pair = build_controlled_fault_pair(
        task,
        render_reference_configuration(task),
    )

    assert pair["regime_version"] == CONTROLLED_FAULT_REGIME_VERSION
    assert pair["baseline_candidate"] == pair["guarded_candidate"]
    assert (
        pair["baseline_candidate_sha256"]
        == pair["guarded_candidate_sha256"]
        == pair["shared_injected_candidate_sha256"]
    )
    assert pair["source_validation"]["valid"] is True
    assert pair["injected_validation"]["valid"] is False


def test_baseline_leaves_injected_candidate_unchanged() -> None:
    task = generate_task(7)
    pair = build_controlled_fault_pair(
        task,
        render_reference_configuration(task),
    )
    result = score_controlled_fault_condition(
        task,
        pair,
        "baseline",
    )

    assert result["repair_applied"] is False
    assert (
        result["final_configuration"]
        == pair["shared_injected_candidate"]
    )
    assert result["validation_before"]["valid"] is False
    assert result["validation_after"]["valid"] is False
    assert result["score"] == 0


def test_guarded_requires_an_explicit_repair_result() -> None:
    task = generate_task(6)
    pair = build_controlled_fault_pair(
        task,
        render_reference_configuration(task),
    )

    with pytest.raises(
        ValueError,
        match="requires one repair result",
    ):
        score_controlled_fault_condition(
            task,
            pair,
            "guarded",
        )


def test_deterministic_rehearsal_repairs_all_default_faults() -> None:
    for index in range(1, 9):
        task = generate_task(index)
        result = run_deterministic_controlled_fault_pair(
            task,
            render_reference_configuration(task),
        )

        assert result["baseline"]["score"] == 0
        assert result["guarded"]["score"] == 1
        assert result["guarded"]["repair_applied"] is True
        assert (
            result["baseline"]["condition_input_candidate_sha256"]
            == result["guarded"][
                "condition_input_candidate_sha256"
            ]
        )


def test_regime_rejects_invalid_source_candidate() -> None:
    task = generate_task(8)
    invalid = generate_direct_candidate(task, 3)

    with pytest.raises(
        ValueError,
        match="requires a valid source candidate",
    ):
        build_controlled_fault_pair(task, invalid)


def test_explicit_fault_class_is_preserved_in_pair_and_scores() -> None:
    task = generate_task(8)
    pair = build_controlled_fault_pair(
        task,
        render_reference_configuration(task),
        fault_class="no_op_command",
    )
    baseline = score_controlled_fault_condition(
        task,
        pair,
        "baseline",
    )

    assert pair["fault_class"] == "no_op_command"
    assert baseline["fault_class"] == "no_op_command"
    assert "NO_OP_COMMAND" in pair["injected_violation_codes"]
