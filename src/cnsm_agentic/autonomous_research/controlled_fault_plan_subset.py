from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .controlled_fault_experiment_plan import (
    load_experiment_plan,
    validate_experiment_plan,
)


SUBSET_PLAN_ID = "controlled_fault_experiment_subset_v1"
SUBSET_PLAN_VERSION = "1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def create_subset_plan(
    parent_plan: dict[str, Any],
    *,
    pair_ids: list[str],
) -> dict[str, Any]:
    """Create a frozen subset while retaining parent-plan provenance."""
    parent_issues = validate_experiment_plan(parent_plan)
    if parent_issues:
        raise ValueError(
            "Parent plan is invalid:\n- " + "\n- ".join(parent_issues)
        )
    if not pair_ids:
        raise ValueError("At least one pair_id is required.")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_ids must be unique.")

    by_id = {
        pair["pair_id"]: pair
        for pair in parent_plan["pairs"]
    }
    missing = [pair_id for pair_id in pair_ids if pair_id not in by_id]
    if missing:
        raise ValueError(
            "Unknown pair IDs: " + ", ".join(sorted(missing))
        )

    selected = [dict(by_id[pair_id]) for pair_id in pair_ids]
    fault_counts = Counter(pair["fault_class"] for pair in selected)
    pattern_counts = Counter(
        pair["workflow_pattern"] for pair in selected
    )

    subset: dict[str, Any] = {
        "plan_type": parent_plan["plan_type"],
        "plan_version": parent_plan["plan_version"],
        "subset_type": SUBSET_PLAN_ID,
        "subset_version": SUBSET_PLAN_VERSION,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "parent_pair_count": parent_plan["pair_count"],
        "selected_pair_ids": list(pair_ids),
        "seed": parent_plan["seed"],
        "pair_count": len(selected),
        "conditions": list(parent_plan["conditions"]),
        "source_generation_calls_per_pair": 1,
        "baseline_repair_calls_per_pair": 0,
        "guarded_repair_calls_per_pair": 1,
        "maximum_model_calls": len(selected) * 2,
        "fault_class_target_count": None,
        "fault_class_counts": dict(sorted(fault_counts.items())),
        "workflow_pattern_target_count": None,
        "workflow_pattern_counts": dict(sorted(pattern_counts.items())),
        "pairs": selected,
    }
    subset["plan_sha256"] = _sha256_json(subset)

    issues = validate_experiment_plan(subset)
    if issues:
        raise RuntimeError(
            "Generated subset failed validation:\n- "
            + "\n- ".join(issues)
        )
    return subset


def write_subset_plan(
    subset: dict[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    issues = validate_experiment_plan(subset)
    if issues:
        raise ValueError(
            "Refusing to write invalid subset:\n- "
            + "\n- ".join(issues)
        )
    if subset.get("subset_type") != SUBSET_PLAN_ID:
        raise ValueError("Missing or unsupported subset_type.")

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(
            subset,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return path


def create_subset_from_file(
    parent_path: Path,
    *,
    pair_ids: list[str],
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    parent = load_experiment_plan(parent_path.resolve())
    subset = create_subset_plan(parent, pair_ids=pair_ids)
    write_subset_plan(subset, output_path, overwrite=overwrite)
    return subset
