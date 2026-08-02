import json
from pathlib import Path

import pytest

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    DEFAULT_PAIR_COUNT,
    compatible_fault_classes,
    generate_experiment_plan,
    load_experiment_plan,
    validate_experiment_plan,
    write_experiment_plan,
)
from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    available_controlled_fault_classes,
    generate_task,
)


def test_default_plan_has_40_unique_pairs() -> None:
    plan = generate_experiment_plan()

    assert plan["pair_count"] == DEFAULT_PAIR_COUNT == 40
    assert plan["maximum_model_calls"] == 80
    assert len(plan["pairs"]) == 40
    assert len({pair["pair_id"] for pair in plan["pairs"]}) == 40
    assert len({pair["task_index"] for pair in plan["pairs"]}) == 40
    assert validate_experiment_plan(plan) == []


def test_default_plan_balances_patterns_and_fault_classes() -> None:
    plan = generate_experiment_plan()

    assert set(plan["fault_class_counts"]) == set(
        available_controlled_fault_classes()
    )
    assert set(plan["fault_class_counts"].values()) == {8}
    assert len(plan["workflow_pattern_counts"]) == 8
    assert set(plan["workflow_pattern_counts"].values()) == {5}


def test_plan_is_deterministic_for_fixed_seed() -> None:
    first = generate_experiment_plan(seed=17)
    second = generate_experiment_plan(seed=17)

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]


def test_different_seed_changes_assignment_but_not_balances() -> None:
    first = generate_experiment_plan(seed=17)
    second = generate_experiment_plan(seed=19)

    first_faults = [pair["fault_class"] for pair in first["pairs"]]
    second_faults = [pair["fault_class"] for pair in second["pairs"]]

    assert first_faults != second_faults
    assert first["fault_class_counts"] == second["fault_class_counts"]
    assert (
        first["workflow_pattern_counts"]
        == second["workflow_pattern_counts"]
    )


def test_every_assignment_is_compatible() -> None:
    plan = generate_experiment_plan()

    for pair in plan["pairs"]:
        task = generate_task(pair["task_index"])
        compatible = compatible_fault_classes(task)
        assert pair["compatible_fault_classes"] == compatible
        assert pair["fault_class"] in compatible


def test_plan_rejects_unbalanced_pair_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        generate_experiment_plan(pair_count=8)

    with pytest.raises(ValueError, match="divisible"):
        generate_experiment_plan(pair_count=25)


def test_validation_rejects_unsupported_assignment() -> None:
    plan = generate_experiment_plan()
    pair = plan["pairs"][0]
    pair["fault_class"] = "not_a_fault"

    issues = validate_experiment_plan(plan)
    assert any("unsupported fault" in issue for issue in issues)
    assert any("plan_sha256" in issue for issue in issues)


def test_write_is_exclusive_and_round_trip_is_valid(
    tmp_path: Path,
) -> None:
    plan = generate_experiment_plan()
    path = tmp_path / "frozen-plan.json"

    written = write_experiment_plan(plan, path)
    assert written == path.resolve()
    assert load_experiment_plan(path) == plan

    with pytest.raises(FileExistsError):
        write_experiment_plan(plan, path)


def test_load_rejects_tampered_plan(tmp_path: Path) -> None:
    plan = generate_experiment_plan()
    path = tmp_path / "tampered.json"
    write_experiment_plan(plan, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["pairs"][0]["task_index"] = 999
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation failed"):
        load_experiment_plan(path)
