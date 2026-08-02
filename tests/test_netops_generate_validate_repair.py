from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    TASK_FAMILY,
    generate_direct_candidate,
    generate_task,
    repair_configuration,
    render_reference_configuration,
    run_condition,
    validate_configuration,
)


def test_generated_task_and_reference_are_valid() -> None:
    task = generate_task(1)
    assert task["task_family"] == TASK_FAMILY
    reference = render_reference_configuration(task)
    validation = validate_configuration(task, reference)
    assert validation["valid"] is True
    assert validation["violation_count"] == 0


def test_validator_detects_wrong_value_missing_setting_and_scope_change() -> None:
    task = generate_task(1)

    wrong = generate_direct_candidate(task, 1)
    wrong_validation = validate_configuration(task, wrong)
    assert wrong_validation["valid"] is False
    assert "INTENT_CONSTRAINT_VIOLATION" in wrong_validation["violation_codes"]

    missing = generate_direct_candidate(task, 2)
    missing_validation = validate_configuration(task, missing)
    assert "MISSING_REQUIRED_SETTING" in missing_validation["violation_codes"]

    scope = generate_direct_candidate(task, 3)
    scope_validation = validate_configuration(task, scope)
    assert "UNINTENDED_INTERFACE_CHANGE" in scope_validation["violation_codes"]


def test_repair_is_bounded_and_produces_valid_configuration() -> None:
    task = generate_task(2)
    candidate = generate_direct_candidate(task, 2)
    result = repair_configuration(task, candidate)
    assert result["repair_applied"] is True
    assert result["validation_before"]["valid"] is False
    assert result["validation_after"]["valid"] is True
    assert result["repaired_configuration"] == render_reference_configuration(task)


def test_repair_does_not_change_already_valid_configuration() -> None:
    task = generate_task(4)
    candidate = render_reference_configuration(task)
    result = repair_configuration(task, candidate)
    assert result["repair_applied"] is False
    assert result["validation_before"]["valid"] is True
    assert result["validation_after"]["valid"] is True


def test_guarded_condition_never_performs_worse_than_direct_candidate() -> None:
    for index in range(1, 9):
        task = generate_task(index)
        baseline = run_condition(task, "baseline", index)
        guarded = run_condition(task, "guarded", index)
        assert guarded["validation_after"]["valid"] is True
        assert int(guarded["validation_after"]["valid"]) >= int(
            baseline["validation_after"]["valid"]
        )


def test_validator_rejects_unsupported_syntax() -> None:
    task = generate_task(1)
    validation = validate_configuration(task, "router ospf 1")
    assert validation["valid"] is False
    assert "SYNTAX_ERROR" in validation["violation_codes"]
