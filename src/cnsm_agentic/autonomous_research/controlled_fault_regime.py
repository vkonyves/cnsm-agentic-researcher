from __future__ import annotations

import hashlib
from typing import Any

from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    inject_controlled_fault,
    repair_configuration,
    validate_configuration,
)


CONTROLLED_FAULT_REGIME_ID = "controlled_fault_repair_regime_v1"
CONTROLLED_FAULT_REGIME_VERSION = "1.0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_controlled_fault_pair(
    task: dict[str, Any],
    valid_source_candidate: str,
    *,
    fault_class: str | None = None,
) -> dict[str, Any]:
    """Create one shared controlled-fault candidate for a paired comparison."""
    source_validation = validate_configuration(
        task,
        valid_source_candidate,
    )
    if not source_validation["valid"]:
        raise ValueError(
            "The controlled-fault regime requires a valid source candidate."
        )

    injection = inject_controlled_fault(
        task,
        valid_source_candidate,
        fault_class=fault_class,
    )
    injected = injection["injected_configuration"]
    injected_hash = _sha256_text(injected)

    return {
        "regime_id": CONTROLLED_FAULT_REGIME_ID,
        "regime_version": CONTROLLED_FAULT_REGIME_VERSION,
        "task_id": task.get("task_id"),
        "fault_class": injection["fault_class"],
        "fault_metadata": injection["fault_metadata"],
        "fault_injector_id": injection["fault_injector_id"],
        "fault_injector_version": injection["fault_injector_version"],
        "source_candidate": valid_source_candidate,
        "source_candidate_sha256": _sha256_text(valid_source_candidate),
        "source_validation": source_validation,
        "shared_injected_candidate": injected,
        "shared_injected_candidate_sha256": injected_hash,
        "injected_validation": injection["injected_validation"],
        "injected_violation_codes": injection[
            "injected_violation_codes"
        ],
        "baseline_candidate": injected,
        "baseline_candidate_sha256": injected_hash,
        "guarded_candidate": injected,
        "guarded_candidate_sha256": injected_hash,
    }


def score_controlled_fault_condition(
    task: dict[str, Any],
    pair: dict[str, Any],
    condition: str,
    *,
    repaired_configuration: str | None = None,
) -> dict[str, Any]:
    """Score one condition while preserving the shared-candidate contract."""
    shared = pair["shared_injected_candidate"]
    shared_hash = pair["shared_injected_candidate_sha256"]

    if condition == "baseline":
        final_configuration = shared
        repair_applied = False
        validation_before = validate_configuration(task, shared)
        validation_after = validation_before
    elif condition == "guarded":
        validation_before = validate_configuration(task, shared)
        if repaired_configuration is None:
            raise ValueError(
                "The guarded controlled-fault condition requires one "
                "repair result."
            )
        final_configuration = repaired_configuration
        repair_applied = final_configuration != shared
        validation_after = validate_configuration(
            task,
            final_configuration,
        )
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    return {
        "regime_id": pair["regime_id"],
        "regime_version": pair["regime_version"],
        "condition": condition,
        "fault_class": pair["fault_class"],
        "shared_injected_candidate": shared,
        "shared_injected_candidate_sha256": shared_hash,
        "condition_input_candidate_sha256": shared_hash,
        "final_configuration": final_configuration,
        "final_configuration_sha256": _sha256_text(
            final_configuration
        ),
        "repair_applied": repair_applied,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "score": int(validation_after["valid"]),
    }


def run_deterministic_controlled_fault_pair(
    task: dict[str, Any],
    valid_source_candidate: str,
    *,
    fault_class: str | None = None,
) -> dict[str, Any]:
    """Development-only rehearsal using the deterministic repairer."""
    pair = build_controlled_fault_pair(
        task,
        valid_source_candidate,
        fault_class=fault_class,
    )
    deterministic_repair = repair_configuration(
        task,
        pair["shared_injected_candidate"],
    )
    baseline = score_controlled_fault_condition(
        task,
        pair,
        "baseline",
    )
    guarded = score_controlled_fault_condition(
        task,
        pair,
        "guarded",
        repaired_configuration=deterministic_repair[
            "repaired_configuration"
        ],
    )
    return {
        "pair": pair,
        "baseline": baseline,
        "guarded": guarded,
        "deterministic_repair": deterministic_repair,
    }
