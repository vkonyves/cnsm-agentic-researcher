from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .netops_generate_validate_repair import (
    available_controlled_fault_classes,
    generate_task,
    inject_controlled_fault,
    render_reference_configuration,
)


EXPERIMENT_PLAN_ID = "controlled_fault_experiment_plan_v1"
EXPERIMENT_PLAN_VERSION = "1.0"
DEFAULT_PAIR_COUNT = 40
DEFAULT_SEED = 17


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


def compatible_fault_classes(task: dict[str, Any]) -> list[str]:
    """Return fault classes that deterministically invalidate this task."""
    reference = render_reference_configuration(task)
    compatible: list[str] = []

    for fault_class in available_controlled_fault_classes():
        try:
            result = inject_controlled_fault(
                task,
                reference,
                fault_class=fault_class,
            )
        except (ValueError, RuntimeError):
            continue

        if result["injected_validation"]["valid"] is False:
            compatible.append(fault_class)

    return compatible


def _balanced_assignment(
    entries: list[dict[str, Any]],
    *,
    target_per_fault: dict[str, int],
    seed: int,
) -> list[str]:
    """Find a deterministic exact-balanced compatible assignment."""
    rng = random.Random(seed)
    classes = sorted(target_per_fault)

    decorated: list[tuple[int, float, int]] = []
    for index, entry in enumerate(entries):
        decorated.append(
            (
                len(entry["compatible_fault_classes"]),
                rng.random(),
                index,
            )
        )
    order = [
        index
        for _, _, index in sorted(decorated)
    ]

    assignments: list[str | None] = [None] * len(entries)
    remaining = dict(target_per_fault)

    def search(position: int) -> bool:
        if position == len(order):
            return all(value == 0 for value in remaining.values())

        entry_index = order[position]
        compatible = [
            fault
            for fault in entries[entry_index]["compatible_fault_classes"]
            if remaining[fault] > 0
        ]
        compatible.sort(
            key=lambda fault: (
                -remaining[fault],
                classes.index(fault),
            )
        )

        for fault in compatible:
            assignments[entry_index] = fault
            remaining[fault] -= 1

            slots_left = len(order) - position - 1
            impossible = any(value > slots_left for value in remaining.values())
            uncovered = any(
                remaining[candidate] > 0
                and not any(
                    candidate
                    in entries[future_index]["compatible_fault_classes"]
                    for future_index in order[position + 1 :]
                )
                for candidate in classes
            )

            if not impossible and not uncovered and search(position + 1):
                return True

            remaining[fault] += 1
            assignments[entry_index] = None

        return False

    if not search(0):
        raise ValueError(
            "No exact balanced fault assignment exists for this task set."
        )

    return [str(value) for value in assignments]


def validate_experiment_plan(plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    if plan.get("plan_type") != EXPERIMENT_PLAN_ID:
        issues.append("Unexpected plan_type.")
    if plan.get("plan_version") != EXPERIMENT_PLAN_VERSION:
        issues.append("Unexpected plan_version.")

    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        issues.append("pairs must be a non-empty list.")
        return sorted(set(issues))

    if plan.get("pair_count") != len(pairs):
        issues.append("pair_count does not match pairs.")

    pair_ids: list[str] = []
    task_indices: list[int] = []
    observed_faults: Counter[str] = Counter()
    observed_patterns: Counter[str] = Counter()

    for position, pair in enumerate(pairs, start=1):
        pair_id = pair.get("pair_id")
        task_index = pair.get("task_index")
        fault_class = pair.get("fault_class")

        if not isinstance(pair_id, str) or not pair_id:
            issues.append(f"Pair {position} has no valid pair_id.")
        else:
            pair_ids.append(pair_id)

        if not isinstance(task_index, int) or isinstance(task_index, bool):
            issues.append(f"Pair {position} has no valid task_index.")
            continue
        task_indices.append(task_index)

        task = generate_task(task_index)
        expected_pattern = task["difficulty"]["pattern"]
        if pair.get("workflow_pattern") != expected_pattern:
            issues.append(
                f"{pair_id} workflow_pattern does not match generated task."
            )

        expected_cycle = (task_index - 1) // 8
        expected_variant = (task_index - 1) % 8
        if pair.get("cycle") != expected_cycle:
            issues.append(f"{pair_id} has an incorrect cycle.")
        if pair.get("variant") != expected_variant:
            issues.append(f"{pair_id} has an incorrect variant.")

        compatible = compatible_fault_classes(task)
        if pair.get("compatible_fault_classes") != compatible:
            issues.append(
                f"{pair_id} compatible fault list is not canonical."
            )
        if fault_class not in compatible:
            issues.append(
                f"{pair_id} uses unsupported fault {fault_class!r}."
            )
        else:
            observed_faults[fault_class] += 1
        observed_patterns[expected_pattern] += 1

    if len(pair_ids) != len(set(pair_ids)):
        issues.append("pair_id values must be unique.")
    if len(task_indices) != len(set(task_indices)):
        issues.append("task_index values must be unique.")

    if dict(sorted(observed_faults.items())) != plan.get(
        "fault_class_counts"
    ):
        issues.append("fault_class_counts do not match pairs.")
    if dict(sorted(observed_patterns.items())) != plan.get(
        "workflow_pattern_counts"
    ):
        issues.append("workflow_pattern_counts do not match pairs.")

    expected_hash = plan.get("plan_sha256")
    hash_payload = dict(plan)
    hash_payload.pop("plan_sha256", None)
    if expected_hash != _sha256_json(hash_payload):
        issues.append("plan_sha256 does not match canonical plan content.")

    return sorted(set(issues))


def generate_experiment_plan(
    *,
    pair_count: int = DEFAULT_PAIR_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not isinstance(pair_count, int) or isinstance(pair_count, bool):
        raise TypeError("pair_count must be an integer.")
    if pair_count <= 0:
        raise ValueError("pair_count must be positive.")

    fault_classes = available_controlled_fault_classes()
    if pair_count % len(fault_classes) != 0:
        raise ValueError(
            "pair_count must be divisible by the number of fault classes "
            "for exact balancing."
        )
    if pair_count % 8 != 0:
        raise ValueError(
            "pair_count must be divisible by 8 for balanced workflow patterns."
        )

    entries: list[dict[str, Any]] = []
    for task_index in range(1, pair_count + 1):
        task = generate_task(task_index)
        compatible = compatible_fault_classes(task)
        if not compatible:
            raise ValueError(
                f"Task {task_index} supports no controlled fault class."
            )
        entries.append({
            "pair_id": f"pair-{task_index:06d}",
            "task_id": f"task-{task_index:06d}",
            "task_index": task_index,
            "cycle": (task_index - 1) // 8,
            "variant": (task_index - 1) % 8,
            "workflow_pattern": task["difficulty"]["pattern"],
            "difficulty_level": task["difficulty"]["level"],
            "fault_class": None,
            "compatible_fault_classes": compatible,
        })

    target = {
        fault: pair_count // len(fault_classes)
        for fault in fault_classes
    }
    assignments = _balanced_assignment(
        entries,
        target_per_fault=target,
        seed=seed,
    )

    for entry, fault_class in zip(entries, assignments):
        entry["fault_class"] = fault_class

    fault_counts = Counter(
        entry["fault_class"] for entry in entries
    )
    pattern_counts = Counter(
        entry["workflow_pattern"] for entry in entries
    )

    plan: dict[str, Any] = {
        "plan_type": EXPERIMENT_PLAN_ID,
        "plan_version": EXPERIMENT_PLAN_VERSION,
        "seed": seed,
        "pair_count": pair_count,
        "conditions": ["baseline", "guarded"],
        "source_generation_calls_per_pair": 1,
        "baseline_repair_calls_per_pair": 0,
        "guarded_repair_calls_per_pair": 1,
        "maximum_model_calls": pair_count * 2,
        "fault_class_target_count": pair_count // len(fault_classes),
        "fault_class_counts": dict(sorted(fault_counts.items())),
        "workflow_pattern_target_count": pair_count // 8,
        "workflow_pattern_counts": dict(sorted(pattern_counts.items())),
        "pairs": entries,
    }
    plan["plan_sha256"] = _sha256_json(plan)

    issues = validate_experiment_plan(plan)
    if issues:
        raise RuntimeError(
            "Generated experiment plan failed validation:\n- "
            + "\n- ".join(issues)
        )
    return plan


def write_experiment_plan(
    plan: dict[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    issues = validate_experiment_plan(plan)
    if issues:
        raise ValueError(
            "Refusing to write invalid experiment plan:\n- "
            + "\n- ".join(issues)
        )

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(
            plan,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return path


def load_experiment_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    issues = validate_experiment_plan(plan)
    if issues:
        raise ValueError(
            "Experiment plan validation failed:\n- "
            + "\n- ".join(issues)
        )
    return plan
