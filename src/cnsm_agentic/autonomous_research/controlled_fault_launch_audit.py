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
from .controlled_fault_regime import build_controlled_fault_pair
from .hosted_controlled_fault_pilot import _repair_prompt, _source_prompt
from .hosted_controlled_fault_plan_runner import (
    PLAN_RUNNER_ID,
    PLAN_RUNNER_VERSION,
    PROMPT_PROTOCOL_VERSION,
    SUPPORTED_PROVIDER,
    _code_fingerprint,
)
from .netops_generate_validate_repair import (
    generate_task,
    render_reference_configuration,
    validate_configuration,
)


AUDIT_ID = "controlled_fault_launch_audit_v1"
AUDIT_VERSION = "1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _output_path_issues(
    *,
    repository_root: Path,
    output_path: Path,
    plan_path: Path,
) -> list[str]:
    issues: list[str] = []
    repository_root = repository_root.resolve()
    output_path = output_path.resolve()
    plan_path = plan_path.resolve()

    if not _is_relative_to(output_path, repository_root):
        issues.append("Audit output must remain inside the repository root.")

    if output_path == plan_path:
        issues.append("Audit output must not overwrite the frozen plan.")

    if output_path.exists() and output_path.is_dir():
        issues.append("Audit output path points to a directory.")

    forbidden_roots = [
        (repository_root / ".git").resolve(),
        (repository_root / "src").resolve(),
        (repository_root / "tests").resolve(),
    ]
    for forbidden in forbidden_roots:
        if output_path == forbidden or _is_relative_to(output_path, forbidden):
            issues.append(
                f"Audit output must not be written under {forbidden}."
            )

    return issues


def audit_controlled_fault_launch(
    *,
    plan_path: Path,
    repository_root: Path,
    output_path: Path,
    model_name: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Audit a frozen plan without creating a provider or making model calls."""
    repository_root = repository_root.resolve()
    plan_path = plan_path.resolve()
    output_path = output_path.resolve()

    issues = _output_path_issues(
        repository_root=repository_root,
        output_path=output_path,
        plan_path=plan_path,
    )

    if max_output_tokens <= 0:
        issues.append("max_output_tokens must be positive.")
    if not model_name.strip():
        issues.append("model_name must be non-empty.")

    plan = load_experiment_plan(plan_path)
    plan_issues = validate_experiment_plan(plan)
    issues.extend(f"Plan: {issue}" for issue in plan_issues)

    expected_calls = (
        int(plan["pair_count"])
        * (
            int(plan["source_generation_calls_per_pair"])
            + int(plan["baseline_repair_calls_per_pair"])
            + int(plan["guarded_repair_calls_per_pair"])
        )
    )
    if expected_calls != int(plan["maximum_model_calls"]):
        issues.append(
            "Plan call ceiling mismatch: "
            f"computed {expected_calls}, "
            f"declared {plan['maximum_model_calls']}."
        )

    pair_reports: list[dict[str, Any]] = []
    source_prompt_hashes: list[str] = []
    repair_prompt_hashes: list[str] = []
    shared_candidate_hashes: list[str] = []
    seen_pair_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    fault_counts: Counter[str] = Counter()
    workflow_counts: Counter[str] = Counter()

    for pair_spec in plan["pairs"]:
        pair_id = str(pair_spec["pair_id"])
        task_id = str(pair_spec["task_id"])
        task_index = int(pair_spec["task_index"])
        fault_class = str(pair_spec["fault_class"])

        pair_issues: list[str] = []
        if pair_id in seen_pair_ids:
            pair_issues.append("Duplicate pair_id.")
        seen_pair_ids.add(pair_id)

        if task_id in seen_task_ids:
            pair_issues.append("Duplicate task_id.")
        seen_task_ids.add(task_id)

        if fault_class not in pair_spec["compatible_fault_classes"]:
            pair_issues.append(
                "Assigned fault is absent from compatible_fault_classes."
            )

        task = generate_task(task_index)
        expected_task_id = f"task-{task_index:06d}"
        if expected_task_id != task_id:
            pair_issues.append(
                f"Plan task_id {task_id!r} does not match "
                f"task_index-derived ID {expected_task_id!r}."
            )

        generated_workflow_pattern = task["difficulty"]["pattern"]
        if generated_workflow_pattern != pair_spec["workflow_pattern"]:
            pair_issues.append(
                "Generated workflow_pattern does not match the plan: "
                f"{generated_workflow_pattern!r} != "
                f"{pair_spec['workflow_pattern']!r}."
            )

        reference = render_reference_configuration(task)
        reference_validation = validate_configuration(task, reference)
        if not reference_validation["valid"]:
            pair_issues.append(
                "Deterministic reference configuration is not valid."
            )

        controlled_pair: dict[str, Any] | None = None
        try:
            controlled_pair = build_controlled_fault_pair(
                task,
                reference,
                fault_class=fault_class,
            )
        except Exception as exc:
            pair_issues.append(
                f"Controlled-fault construction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        source_prompt = _source_prompt(task)
        source_prompt_hash = _sha256_text(source_prompt)
        source_prompt_hashes.append(source_prompt_hash)

        repair_prompt_hash: str | None = None
        shared_candidate_hash: str | None = None
        injected_violation_codes: list[str] = []

        if controlled_pair is not None:
            shared_candidate = controlled_pair["shared_injected_candidate"]
            injected_validation = controlled_pair["injected_validation"]

            shared_candidate_hash = _sha256_text(shared_candidate)
            shared_candidate_hashes.append(shared_candidate_hash)

            if injected_validation["valid"]:
                pair_issues.append(
                    "Injected candidate unexpectedly remains valid."
                )

            injected_violation_codes = sorted(
                str(code)
                for code in injected_validation.get(
                    "violation_codes", []
                )
            )
            if not injected_violation_codes:
                pair_issues.append(
                    "Injected candidate has no validator violation codes."
                )

            repair_prompt = _repair_prompt(
                task,
                shared_candidate,
                injected_validation,
            )
            repair_prompt_hash = _sha256_text(repair_prompt)
            repair_prompt_hashes.append(repair_prompt_hash)

            rebuilt = build_controlled_fault_pair(
                task,
                reference,
                fault_class=fault_class,
            )
            if rebuilt != controlled_pair:
                pair_issues.append(
                    "Controlled-fault construction is not deterministic."
                )

            if shared_candidate != rebuilt["shared_injected_candidate"]:
                pair_issues.append(
                    "Shared injected candidate changed across rebuilds."
                )

        metadata_preview = {
            "source": {
                "pair_id": pair_id,
                "task_id": task_id,
                "task_index": str(task_index),
                "stage": "valid_source_generation",
                "plan_sha256": plan["plan_sha256"],
            },
            "repair": {
                "pair_id": pair_id,
                "task_id": task_id,
                "task_index": str(task_index),
                "stage": "controlled_fault_repair",
                "fault_class": fault_class,
                "plan_sha256": plan["plan_sha256"],
            },
        }
        for stage, metadata in metadata_preview.items():
            for key, value in metadata.items():
                if not isinstance(value, str):
                    pair_issues.append(
                        f"{stage} metadata {key!r} is not a string."
                    )

        fault_counts[fault_class] += 1
        workflow_counts[str(pair_spec["workflow_pattern"])] += 1

        pair_reports.append(
            {
                "pair_id": pair_id,
                "task_id": task_id,
                "task_index": task_index,
                "workflow_pattern": pair_spec["workflow_pattern"],
                "fault_class": fault_class,
                "reference_configuration_sha256": _sha256_text(reference),
                "source_prompt_sha256": source_prompt_hash,
                "repair_prompt_sha256": repair_prompt_hash,
                "shared_injected_candidate_sha256": shared_candidate_hash,
                "injected_violation_codes": injected_violation_codes,
                "metadata_preview": metadata_preview,
                "issues": pair_issues,
                "status": "PASS" if not pair_issues else "FAIL",
            }
        )
        issues.extend(f"{pair_id}: {issue}" for issue in pair_issues)

    aggregate_prompt_payload = {
        "source_prompt_hashes": source_prompt_hashes,
        "repair_prompt_hashes": repair_prompt_hashes,
    }
    aggregate_candidate_payload = {
        "shared_injected_candidate_hashes": shared_candidate_hashes,
    }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "audit_version": AUDIT_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "provider_calls_made": 0,
        "provider_initialized": False,
        "repository_root": str(repository_root),
        "plan_path": str(plan_path),
        "output_path": str(output_path),
        "plan_sha256": plan["plan_sha256"],
        "pair_count": plan["pair_count"],
        "maximum_model_calls": plan["maximum_model_calls"],
        "computed_maximum_model_calls": expected_calls,
        "model_settings": {
            "provider": SUPPORTED_PROVIDER,
            "model_name": model_name,
            "max_output_tokens": max_output_tokens,
            "maximum_attempts_per_call": 1,
            "reasoning_effort": "minimal",
        },
        "runner": {
            "runner_id": PLAN_RUNNER_ID,
            "runner_version": PLAN_RUNNER_VERSION,
            "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
            "code_fingerprint_sha256": _code_fingerprint(),
        },
        "fault_class_counts": dict(sorted(fault_counts.items())),
        "workflow_pattern_counts": dict(sorted(workflow_counts.items())),
        "source_prompt_count": len(source_prompt_hashes),
        "repair_prompt_count": len(repair_prompt_hashes),
        "aggregate_prompt_sha256": _sha256_json(
            aggregate_prompt_payload
        ),
        "aggregate_shared_candidate_sha256": _sha256_json(
            aggregate_candidate_payload
        ),
        "pairs": pair_reports,
        "issues": issues,
    }
    report["audit_report_sha256"] = _sha256_json(report)
    return report


def write_launch_audit(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return output_path
