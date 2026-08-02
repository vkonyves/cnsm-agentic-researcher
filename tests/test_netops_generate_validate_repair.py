import pytest

from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    FAULT_INJECTOR_VERSION,
    TASK_GENERATOR_VERSION,
    VALIDATOR_VERSION,
    available_controlled_fault_classes,
    generate_direct_candidate,
    generate_task,
    inject_controlled_fault,
    repair_configuration,
    render_reference_configuration,
    run_condition,
    validate_configuration,
)


def test_reference_workflows_are_valid_for_all_patterns() -> None:
    for index in range(1, 9):
        task = generate_task(index)
        result = validate_configuration(
            task,
            render_reference_configuration(task),
        )
        assert result["valid"] is True, (index, result["violations"])
        assert task["task_generator_version"] == TASK_GENERATOR_VERSION
        assert result["validator_version"] == VALIDATOR_VERSION


def test_task_bank_has_eight_sequence_patterns() -> None:
    tasks = [generate_task(index) for index in range(1, 9)]
    assert len({task["difficulty"]["pattern"] for task in tasks}) == 8
    assert {task["difficulty"]["level"] for task in tasks} == {
        "medium",
        "hard",
        "very_hard",
        "extreme",
    }
    assert max(
        task["difficulty"]["required_command_count"]
        for task in tasks
    ) == 9


def test_controlled_fault_defaults_invalidate_all_task_patterns() -> None:
    observed_faults = set()
    for index in range(1, 9):
        task = generate_task(index)
        source = render_reference_configuration(task)
        result = inject_controlled_fault(task, source)

        assert result["fault_injector_version"] == FAULT_INJECTOR_VERSION
        assert result["source_validation"]["valid"] is True
        assert result["injected_validation"]["valid"] is False
        assert result["injected_violation_codes"]
        assert result["source_configuration"] != result[
            "injected_configuration"
        ]
        observed_faults.add(result["fault_class"])

    assert observed_faults == {
        "offline_change_before_shutdown",
        "break_before_make",
        "dropped_required_restore",
        "protected_interface_change",
    }


@pytest.mark.parametrize(
    "fault_class, expected_code",
    [
        (
            "offline_change_before_shutdown",
            "INTERFACE_NOT_DOWN_FOR_CHANGE",
        ),
        (
            "protected_interface_change",
            "PROTECTED_INTERFACE_CHANGE",
        ),
        (
            "no_op_command",
            "NO_OP_COMMAND",
        ),
    ],
)
def test_explicit_fault_classes_have_expected_violations(
    fault_class: str,
    expected_code: str,
) -> None:
    task = generate_task(8)
    result = inject_controlled_fault(
        task,
        render_reference_configuration(task),
        fault_class=fault_class,
    )
    assert expected_code in result["injected_violation_codes"]


def test_break_before_make_creates_transient_availability_failure() -> None:
    task = generate_task(2)
    result = inject_controlled_fault(
        task,
        render_reference_configuration(task),
        fault_class="break_before_make",
    )
    assert (
        "TRANSIENT_AVAILABILITY_VIOLATION"
        in result["injected_violation_codes"]
    )


def test_dropped_restore_creates_final_state_failure() -> None:
    task = generate_task(7)
    result = inject_controlled_fault(
        task,
        render_reference_configuration(task),
        fault_class="dropped_required_restore",
    )
    assert "FINAL_STATE_MISMATCH" in result["injected_violation_codes"]


def test_controlled_fault_rejects_invalid_source_candidate() -> None:
    task = generate_task(8)
    invalid = generate_direct_candidate(task, 3)
    assert validate_configuration(task, invalid)["valid"] is False
    with pytest.raises(
        ValueError,
        match="only be injected into valid configurations",
    ):
        inject_controlled_fault(task, invalid)


def test_available_fault_classes_are_stable_and_unique() -> None:
    classes = available_controlled_fault_classes()
    assert classes == [
        "offline_change_before_shutdown",
        "break_before_make",
        "dropped_required_restore",
        "protected_interface_change",
        "no_op_command",
    ]
    assert len(classes) == len(set(classes))


def test_injected_candidate_is_repairable_by_bounded_repair() -> None:
    task = generate_task(8)
    injected = inject_controlled_fault(
        task,
        render_reference_configuration(task),
    )
    repaired = repair_configuration(
        task,
        injected["injected_configuration"],
    )
    assert repaired["repair_applied"] is True
    assert repaired["validation_before"]["valid"] is False
    assert repaired["validation_after"]["valid"] is True


def test_shutdown_is_required_before_vlan_or_mtu_change() -> None:
    task = generate_task(1)
    unsafe = "\n".join(
        [
            "interface edge1 mtu 1600",
            "interface edge1 admin down",
            "interface edge1 vlan 30",
            "interface edge1 admin up",
        ]
    )
    result = validate_configuration(task, unsafe)
    assert result["valid"] is False
    assert "INTERFACE_NOT_DOWN_FOR_CHANGE" in result["violation_codes"]


def test_make_before_break_preserves_transient_availability() -> None:
    task = generate_task(2)
    unsafe = "\n".join(
        [
            "interface uplink1 admin down",
            "interface uplink2 admin up",
        ]
    )
    result = validate_configuration(task, unsafe)
    assert result["valid"] is False
    assert "TRANSIENT_AVAILABILITY_VIOLATION" in result["violation_codes"]


def test_peer_group_must_be_quiesced_before_atomic_mtu_change() -> None:
    task = generate_task(3)
    unsafe = "\n".join(
        [
            "interface edge1 admin down",
            "interface edge1 mtu 1800",
            "interface edge2 admin down",
            "interface edge2 mtu 1800",
            "interface edge1 admin up",
            "interface edge2 admin up",
        ]
    )
    result = validate_configuration(task, unsafe)
    assert result["valid"] is False
    assert "PEER_GROUP_NOT_QUIESCED" in result["violation_codes"]


def test_protected_management_interface_cannot_change() -> None:
    task = generate_task(1)
    candidate = (
        render_reference_configuration(task)
        + "\ninterface mgmt1 admin down"
    )
    result = validate_configuration(task, candidate)
    assert "PROTECTED_INTERFACE_CHANGE" in result["violation_codes"]
    assert "UNINTENDED_STATE_CHANGE" in result["violation_codes"]


def test_no_op_commands_are_rejected() -> None:
    task = generate_task(2)
    candidate = (
        "interface uplink1 admin up\n"
        + render_reference_configuration(task)
    )
    result = validate_configuration(task, candidate)
    assert "NO_OP_COMMAND" in result["violation_codes"]


def test_controlled_candidates_cover_value_missing_and_sequence_failures() -> None:
    task = generate_task(8)

    wrong = validate_configuration(
        task,
        generate_direct_candidate(task, 1),
    )
    assert wrong["valid"] is False
    assert "FINAL_STATE_MISMATCH" in wrong["violation_codes"]

    missing = validate_configuration(
        task,
        generate_direct_candidate(task, 2),
    )
    assert missing["valid"] is False
    assert "FINAL_STATE_MISMATCH" in missing["violation_codes"]

    unsafe = validate_configuration(
        task,
        generate_direct_candidate(task, 3),
    )
    assert unsafe["valid"] is False
    assert any(
        code in unsafe["violation_codes"]
        for code in {
            "INTERFACE_NOT_DOWN_FOR_CHANGE",
            "TRANSIENT_AVAILABILITY_VIOLATION",
        }
    )


def test_repair_restores_valid_ordered_workflow() -> None:
    task = generate_task(7)
    candidate = generate_direct_candidate(task, 3)
    repaired = repair_configuration(task, candidate)
    assert repaired["repair_applied"] is True
    assert repaired["validation_before"]["valid"] is False
    assert repaired["validation_after"]["valid"] is True


def test_guarded_condition_is_monotonic_across_two_cycles() -> None:
    for index in range(1, 17):
        task = generate_task(index)
        baseline = run_condition(task, "baseline", index)
        guarded = run_condition(task, "guarded", index)
        assert guarded["validation_after"]["valid"] is True
        assert int(guarded["validation_after"]["valid"]) >= int(
            baseline["validation_after"]["valid"]
        )


def test_unknown_interface_and_syntax_are_rejected() -> None:
    task = generate_task(1)
    syntax = validate_configuration(task, "router ospf 1")
    assert "SYNTAX_ERROR" in syntax["violation_codes"]

    unknown = validate_configuration(
        task,
        render_reference_configuration(task)
        + "\ninterface ghost0 admin down",
    )
    assert "UNKNOWN_INTERFACE" in unknown["violation_codes"]
