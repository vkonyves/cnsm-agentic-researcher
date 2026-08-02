from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    TASK_FAMILY,
    TASK_GENERATOR_VERSION,
    VALIDATOR_VERSION,
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
    assert task["task_generator_version"] == TASK_GENERATOR_VERSION
    assert len(task["initial_state"]) == 4
    reference = render_reference_configuration(task)
    validation = validate_configuration(task, reference)
    assert validation["valid"] is True
    assert validation["validator_version"] == VALIDATOR_VERSION
    assert validation["violation_count"] == 0
    assert validation["difficulty"] == task["difficulty"]


def test_task_bank_has_eight_distinct_difficulty_patterns() -> None:
    tasks = [generate_task(index) for index in range(1, 9)]
    patterns = {task["difficulty"]["pattern"] for task in tasks}
    assert len(patterns) == 8
    assert {task["difficulty"]["level"] for task in tasks} == {
        "medium",
        "hard",
        "very_hard",
    }
    assert max(
        task["difficulty"]["required_assignment_count"]
        for task in tasks
    ) == 4
    assert max(
        task["difficulty"]["changed_interface_count"]
        for task in tasks
    ) == 3


def test_task_variants_include_multi_interface_and_preservation_cases() -> None:
    task_two = generate_task(2)
    changed_interfaces = {
        item["interface"] for item in task_two["required_changes"]
    }
    assert len(changed_interfaces) == 2
    assert task_two["constraints"]["preserve_unspecified_state"] is True

    task_five = generate_task(5)
    assert task_five["difficulty"]["changed_interface_count"] == 3
    assert "Preserve all VLANs" in task_five["intent"]

    task_eight = generate_task(8)
    assert len(task_eight["required_changes"]) == 4
    assert "preserving admin up and MTU 9000" in task_eight["intent"]


def test_validator_detects_wrong_value_missing_setting_and_preserved_change() -> None:
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
    assert "UNINTENDED_STATE_CHANGE" in scope_validation["violation_codes"]


def test_validator_rejects_change_to_unspecified_field_on_target_interface() -> None:
    task = generate_task(6)
    reference = render_reference_configuration(task)
    target = task["required_changes"][0]["interface"]
    candidate = reference + f"\ninterface {target} mtu 1700"
    validation = validate_configuration(task, candidate)
    assert validation["valid"] is False
    assert "UNINTENDED_STATE_CHANGE" in validation["violation_codes"]


def test_repair_is_bounded_and_produces_valid_configuration() -> None:
    task = generate_task(7)
    candidate = generate_direct_candidate(task, 2)
    result = repair_configuration(task, candidate)
    assert result["repair_applied"] is True
    assert result["validation_before"]["valid"] is False
    assert result["validation_after"]["valid"] is True
    assert result["repaired_configuration"] == render_reference_configuration(task)


def test_repair_does_not_change_already_valid_configuration() -> None:
    task = generate_task(8)
    candidate = render_reference_configuration(task)
    result = repair_configuration(task, candidate)
    assert result["repair_applied"] is False
    assert result["validation_before"]["valid"] is True
    assert result["validation_after"]["valid"] is True


def test_guarded_condition_never_performs_worse_than_direct_candidate() -> None:
    for index in range(1, 17):
        task = generate_task(index)
        baseline = run_condition(task, "baseline", index)
        guarded = run_condition(task, "guarded", index)
        assert guarded["validation_after"]["valid"] is True
        assert int(guarded["validation_after"]["valid"]) >= int(
            baseline["validation_after"]["valid"]
        )


def test_validator_rejects_unsupported_syntax_and_unknown_interface() -> None:
    task = generate_task(1)
    syntax = validate_configuration(task, "router ospf 1")
    assert syntax["valid"] is False
    assert "SYNTAX_ERROR" in syntax["violation_codes"]

    unknown = validate_configuration(
        task,
        render_reference_configuration(task) + "\ninterface ghost0 admin down",
    )
    assert unknown["valid"] is False
    assert "UNKNOWN_INTERFACE" in unknown["violation_codes"]
