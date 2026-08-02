from pathlib import Path

import pytest

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    generate_experiment_plan,
    load_experiment_plan,
)
from cnsm_agentic.autonomous_research.controlled_fault_plan_subset import (
    SUBSET_PLAN_ID,
    create_subset_plan,
    write_subset_plan,
)


def test_subset_preserves_selected_pairs_and_parent_hash() -> None:
    parent = generate_experiment_plan(pair_count=40, seed=17)
    subset = create_subset_plan(
        parent,
        pair_ids=["pair-000002", "pair-000008"],
    )

    assert subset["subset_type"] == SUBSET_PLAN_ID
    assert subset["parent_plan_sha256"] == parent["plan_sha256"]
    assert subset["selected_pair_ids"] == [
        "pair-000002",
        "pair-000008",
    ]
    assert subset["pair_count"] == 2
    assert subset["maximum_model_calls"] == 4
    assert [pair["pair_id"] for pair in subset["pairs"]] == [
        "pair-000002",
        "pair-000008",
    ]


def test_subset_is_deterministic() -> None:
    parent = generate_experiment_plan(pair_count=40, seed=17)
    first = create_subset_plan(
        parent,
        pair_ids=["pair-000002", "pair-000008"],
    )
    second = create_subset_plan(
        parent,
        pair_ids=["pair-000002", "pair-000008"],
    )

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]


def test_subset_rejects_unknown_or_duplicate_ids() -> None:
    parent = generate_experiment_plan(pair_count=40, seed=17)

    with pytest.raises(ValueError, match="Unknown pair IDs"):
        create_subset_plan(parent, pair_ids=["pair-999999"])

    with pytest.raises(ValueError, match="must be unique"):
        create_subset_plan(
            parent,
            pair_ids=["pair-000002", "pair-000002"],
        )


def test_subset_round_trip_is_valid(tmp_path: Path) -> None:
    parent = generate_experiment_plan(pair_count=40, seed=17)
    subset = create_subset_plan(
        parent,
        pair_ids=["pair-000002", "pair-000008"],
    )
    path = tmp_path / "subset.json"
    write_subset_plan(subset, path)

    loaded = load_experiment_plan(path)
    assert loaded == subset

    with pytest.raises(FileExistsError):
        write_subset_plan(subset, path)
