from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controlled_fault_experiment_plan import generate_experiment_plan
from .controlled_fault_regime import build_controlled_fault_pair
from .execution_adapters import (
    COMPLETED_STATUS,
    PAIRED_BINARY_REQUIRED_ROW_FIELDS,
    PAIRED_BINARY_RESULT_SCHEMA_ID,
    PAIRED_BINARY_RESULT_SCHEMA_VERSION,
    SUPPORTED_PAIRED_CONDITIONS,
    _canonical_json_bytes,
    _relative_to_parent,
    _sha256_bytes,
    _sha256_file,
    _write_json,
    validate_paired_binary_result_row,
)
from .model_providers import (
    HostedModelProvider,
    JsonFileCallCache,
    ModelCallRequest,
    OpenAIResponsesProvider,
)
from .netops_generate_validate_repair import (
    TASK_FAMILY,
    TASK_GENERATOR_ID,
    TASK_GENERATOR_VERSION,
    VALIDATOR_ID,
    VALIDATOR_VERSION,
    generate_task,
    render_reference_configuration,
    validate_configuration,
)

SUPPORTED_PROVIDER = "openai_responses"
FAULT_ASSIGNMENT_SEED = 17


def _state_lines(task: dict[str, Any]) -> str:
    return "\n".join(
        f"- {name}: admin={state['admin']}, "
        f"mtu={state['mtu']}, vlan={state['vlan']}"
        for name, state in task["initial_state"].items()
    )


def _source_prompt(task: dict[str, Any]) -> str:
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Current state:\n"
        + _state_lines(task)
        + "\nReturn a valid ordered command sequence. Transient safety "
        "constraints must hold after every command. Generate only required "
        "commands; do not restate preserved settings or add no-op commands. "
        "Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>\n"
        "Do not use Markdown or explanatory prose."
    )


def _blind_repair_prompt(
    task: dict[str, Any],
    injected_candidate: str,
) -> str:
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Initial state:\n"
        + _state_lines(task)
        + "\nThe candidate contains one operational defect.\n"
        f"Candidate:\n{injected_candidate}\n"
        "Return one corrected complete ordered sequence. Preserve all unrelated "
        "settings, satisfy transient constraints after every command, and do not "
        "include no-op or explanatory text. Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>"
    )


def _guided_repair_prompt(
    task: dict[str, Any],
    injected_candidate: str,
    validation: dict[str, Any],
) -> str:
    violations = "\n".join(
        f"- {item['code']}: {item['message']}"
        for item in validation["violations"]
    )
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Initial state:\n"
        + _state_lines(task)
        + "\nThe candidate contains one operational defect and failed "
        "deterministic validation.\n"
        f"Candidate:\n{injected_candidate}\n"
        f"Validation feedback:\n{violations}\n"
        "Return one corrected complete ordered sequence. Preserve all unrelated "
        "settings, satisfy transient constraints after every command, and do not "
        "include no-op or explanatory text. Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>"
    )


def _provider_for(
    *,
    provider: HostedModelProvider | None,
    output_dir: Path,
    plan: dict[str, Any],
) -> HostedModelProvider:
    if provider is not None:
        return provider
    return OpenAIResponsesProvider(
        cache=JsonFileCallCache(output_dir / "call_cache"),
        maximum_attempts=int(plan["maximum_attempts_per_call"]),
        retry_backoff_seconds=float(plan.get("retry_backoff_seconds", 1.0)),
    )


def _call_status(*calls: dict[str, Any] | None) -> str:
    actual = [call for call in calls if call is not None]
    if actual and all(call.get("cache_status") == "HIT" for call in actual):
        return "CACHED"
    return "COMPLETED"


def execute_validator_feedback_study(
    *,
    plan: dict[str, Any],
    preregistration: dict[str, Any],
    output_dir: Path,
    provider: HostedModelProvider | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    scoring_dir = output_dir / "scoring"
    provider_dir = output_dir / "provider_calls"
    fault_dir = output_dir / "faults"
    for path in (responses_dir, scoring_dir, provider_dir, fault_dir):
        path.mkdir(exist_ok=True)

    study_id = str(plan["study_id"])
    task_count = int(plan["task_count"])
    started_at = datetime.now(timezone.utc)
    hosted = _provider_for(
        provider=provider,
        output_dir=output_dir,
        plan=plan,
    )

    # Reuse the already-tested deterministic balancing algorithm, but archive a
    # v2 assignment artifact rather than the old v1 treatment semantics.
    legacy_balance = generate_experiment_plan(
        pair_count=task_count,
        seed=FAULT_ASSIGNMENT_SEED,
    )
    assignment = {
        "schema_version": "2.0",
        "assignment_id": "balanced_controlled_fault_assignment_v2",
        "seed": FAULT_ASSIGNMENT_SEED,
        "selection_is_outcome_independent": True,
        "pair_count": task_count,
        "fault_class_counts": legacy_balance["fault_class_counts"],
        "workflow_pattern_counts": legacy_balance["workflow_pattern_counts"],
        "pairs": legacy_balance["pairs"],
        "balancing_algorithm_source_plan_sha256": legacy_balance["plan_sha256"],
    }
    assignment["assignment_sha256"] = _sha256_bytes(
        _canonical_json_bytes(assignment)
    )
    assignment_path = output_dir / "controlled_fault_assignment.json"
    _write_json(assignment_path, assignment)
    assignment_hash = _sha256_file(assignment_path)
    fault_by_index = {
        int(item["task_index"]): item
        for item in assignment["pairs"]
    }

    tasks: list[dict[str, Any]] = []
    for index in plan["task_indices"]:
        payload = generate_task(index)
        task_id = f"task-{index:06d}"
        pair_id = f"pair-{index:06d}"
        reference = render_reference_configuration(payload)
        tasks.append({
            "schema_version": "2.0",
            "task_manifest_id": "task-manifest-v2",
            "study_id": study_id,
            "task_id": task_id,
            "pair_id": pair_id,
            "task_index": index,
            "task_family": TASK_FAMILY,
            "generator_id": TASK_GENERATOR_ID,
            "generator_version": TASK_GENERATOR_VERSION,
            "generation_seed": index,
            "source_identifier": f"synthetic-netops:{task_id}",
            "task_payload": payload,
            "reference_answer": reference,
            "task_input_sha256": _sha256_bytes(
                _canonical_json_bytes(payload)
            ),
            "reference_answer_sha256": _sha256_bytes(
                reference.encode("utf-8")
            ),
            "contamination_checks": [],
        })

    task_manifest_path = output_dir / "task_manifest.jsonl"
    task_manifest_path.write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )
    task_manifest_hash = _sha256_file(task_manifest_path)

    transformation_manifest = {
        "schema_version": "2.0",
        "study_id": study_id,
        "task_family": TASK_FAMILY,
        "generation_semantics": "shared_controlled_fault_candidate",
        "controlled_fault_assignment_sha256": assignment["assignment_sha256"],
        "repair_order_rule": (
            "odd task_index: baseline then guarded; "
            "even task_index: guarded then baseline"
        ),
        "repair_order_is_outcome_independent": True,
        "conditions": {
            "baseline": {
                "transformation_id": plan["transformations"]["baseline"],
                "scientific_label": "blind_one_shot_repair",
                "validator_diagnostics_exposed": False,
                "maximum_repair_calls": 1,
            },
            "guarded": {
                "transformation_id": plan["transformations"]["guarded"],
                "scientific_label": "validator_guided_one_shot_repair",
                "validator_diagnostics_exposed": True,
                "maximum_repair_calls": 1,
            },
        },
    }
    transformation_path = output_dir / "transformation_manifest.json"
    _write_json(transformation_path, transformation_manifest)
    transformation_hash = _sha256_file(transformation_path)

    model_configuration = {
        "provider": SUPPORTED_PROVIDER,
        "requested_model": plan["model_name"],
        "declared_model_version": plan["model_version"],
        "maximum_attempts_per_call": 1,
        "max_output_tokens": plan["max_output_tokens"],
        "temperature": plan.get("temperature"),
        "reasoning_effort": plan["reasoning_effort"],
        "store": False,
        "execution_mode": str(plan["execution_mode"]),
    }
    model_configuration_path = output_dir / "model_configuration.json"
    _write_json(model_configuration_path, model_configuration)
    model_configuration_hash = _sha256_file(model_configuration_path)

    result_schema_path = output_dir / "result_schema.json"
    _write_json(result_schema_path, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": PAIRED_BINARY_RESULT_SCHEMA_ID,
        "schema_version": PAIRED_BINARY_RESULT_SCHEMA_VERSION,
        "type": "object",
        "required": list(PAIRED_BINARY_REQUIRED_ROW_FIELDS),
    })

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    model_calls_used = 0
    source_valid_count = 0

    for task_record in tasks:
        payload = task_record["task_payload"]
        task_id = task_record["task_id"]
        pair_id = task_record["pair_id"]
        task_index = int(task_record["task_index"])
        task_started = datetime.now(timezone.utc)

        source_prompt = _source_prompt(payload)
        source_request = ModelCallRequest(
            provider=SUPPORTED_PROVIDER,
            model=str(plan["model_name"]),
            prompt=source_prompt,
            instructions="Generate one valid bounded NetOps workflow.",
            temperature=plan.get("temperature"),
            max_output_tokens=int(plan["max_output_tokens"]),
            reasoning_effort=str(plan["reasoning_effort"]),
            metadata={
                "study_id": study_id,
                "task_id": task_id,
                "pair_id": pair_id,
                "task_index": str(task_index),
                "stage": "valid_source_generation",
            },
        )

        source_call: dict[str, Any] | None = None
        source_candidate: str | None = None
        source_validation: dict[str, Any] | None = None
        source_error_type: str | None = None
        source_error_message: str | None = None
        source_attempts = 0
        try:
            result = hosted.call(source_request)
            source_call = result.to_dict()
            source_candidate = result.response_text.strip()
            source_validation = validate_configuration(
                payload,
                source_candidate,
            )
            source_attempts = int(result.attempt_count)
        except Exception as exc:
            source_error_type = type(exc).__name__
            source_error_message = str(exc)
            source_attempts = 1

        model_calls_used += source_attempts
        if model_calls_used > int(plan["maximum_model_calls"]):
            raise RuntimeError(
                "Validator-feedback adapter exceeded frozen model-call ceiling."
            )

        source_provider_path = (
            provider_dir / f"{task_id}-source-generation.json"
        )
        _write_json(source_provider_path, {
            "schema_version": "2.0",
            "stage": "valid_source_generation",
            "call": source_call,
            "terminal_error_type": source_error_type,
            "terminal_error_message": source_error_message,
        })
        source_provider_relative = _relative_to_parent(
            source_provider_path,
            output_dir,
        )
        source_provider_hash = _sha256_file(source_provider_path)

        source_response_path: Path | None = None
        source_response_hash: str | None = None
        if source_candidate is not None:
            source_response_path = (
                responses_dir / f"{task_id}-valid-source.txt"
            )
            source_response_path.write_text(
                source_candidate + "\n",
                encoding="utf-8",
            )
            source_response_hash = _sha256_file(source_response_path)

        events.append({
            "event_type": "hosted_model_call",
            "study_id": study_id,
            "task_id": task_id,
            "pair_id": pair_id,
            "condition": "shared",
            "stage": "valid_source_generation",
            "outcome": (
                "COMPLETED" if source_call is not None else "FAILED"
            ),
            "resolved_model": (
                source_call.get("resolved_model")
                if source_call is not None
                else None
            ),
            "model_calls_used": source_attempts,
        })

        eligible = bool(
            source_candidate is not None
            and source_validation is not None
            and source_validation["valid"]
        )

        if not eligible:
            for order, condition in enumerate(
                SUPPORTED_PAIRED_CONDITIONS,
                1,
            ):
                episode_id = f"{task_id}-{condition}"
                reason = (
                    "SOURCE_CALL_FAILED"
                    if source_candidate is None
                    else "INVALID_SOURCE_CANDIDATE"
                )
                scoring_record = {
                    "schema_version": "2.0",
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "condition": condition,
                    "scoring_status": "NOT_SCORED",
                    "score": None,
                    "score_reason_code": reason,
                    "source_validation": source_validation,
                    "controlled_challenge_eligible": False,
                }
                scoring_path = scoring_dir / f"{episode_id}.json"
                _write_json(scoring_path, scoring_record)
                scoring_relative = _relative_to_parent(
                    scoring_path,
                    output_dir,
                )
                scoring_hash = _sha256_file(scoring_path)

                call_status = (
                    "FAILED" if source_candidate is None else _call_status(source_call)
                )
                row = {
                    "schema_version": "1.0",
                    "result_schema_id": PAIRED_BINARY_RESULT_SCHEMA_ID,
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "task_family": TASK_FAMILY,
                    "condition": condition,
                    "paired_condition": (
                        "guarded" if condition == "baseline" else "baseline"
                    ),
                    "condition_order": order,
                    "execution_mode": str(plan["execution_mode"]),
                    "model_provider": SUPPORTED_PROVIDER,
                    "model_name": str(plan["model_name"]),
                    "model_version": (
                        str(source_call.get("resolved_model"))
                        if source_call is not None
                        and source_call.get("resolved_model")
                        else str(plan["model_version"])
                    ),
                    "model_configuration_sha256": model_configuration_hash,
                    "task_manifest_id": "task-manifest-v2",
                    "task_manifest_sha256": task_manifest_hash,
                    "task_input_sha256": task_record["task_input_sha256"],
                    "reference_answer_sha256": task_record[
                        "reference_answer_sha256"
                    ],
                    "transformation_id": plan["transformations"][condition],
                    "transformation_manifest_sha256": transformation_hash,
                    "prompt_sha256": _sha256_bytes(
                        source_prompt.encode("utf-8")
                    ),
                    "call_status": call_status,
                    "attempt_count": max(1, source_attempts),
                    "model_calls_used": source_attempts,
                    "terminal_error_type": (
                        source_error_type if call_status == "FAILED" else None
                    ),
                    "terminal_error_message": (
                        source_error_message if call_status == "FAILED" else None
                    ),
                    "response_sha256": (
                        source_response_hash
                        if call_status in {"COMPLETED", "CACHED"}
                        else None
                    ),
                    "response_artifact_path": (
                        _relative_to_parent(source_response_path, output_dir)
                        if source_response_path is not None
                        and call_status in {"COMPLETED", "CACHED"}
                        else None
                    ),
                    "scoring_status": "NOT_SCORED",
                    "score": None,
                    "score_reason_code": reason,
                    "scorer_id": VALIDATOR_ID,
                    "scorer_version": VALIDATOR_VERSION,
                    "scoring_input_sha256": _sha256_bytes(
                        _canonical_json_bytes(scoring_record)
                    ),
                    "scoring_artifact_path": scoring_relative,
                    "scoring_artifact_sha256": scoring_hash,
                    "contamination_flags": [],
                    "validity_flags": [
                        "CONTROLLED_CHALLENGE_INELIGIBLE"
                    ],
                    "started_at_utc": task_started.isoformat(),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": 0,
                }
                row_issues = validate_paired_binary_result_row(row)
                if row_issues:
                    raise RuntimeError(
                        "Generated invalid ineligible paired row: "
                        + "; ".join(row_issues)
                    )
                rows.append(row)
            continue

        source_valid_count += 1
        fault_spec = fault_by_index[task_index]
        pair = build_controlled_fault_pair(
            payload,
            source_candidate,
            fault_class=str(fault_spec["fault_class"]),
        )
        injected = pair["shared_injected_candidate"]
        injected_path = (
            responses_dir / f"{task_id}-shared-controlled-fault.txt"
        )
        injected_path.write_text(injected + "\n", encoding="utf-8")
        injected_hash = _sha256_file(injected_path)

        fault_path = fault_dir / f"{task_id}-fault.json"
        _write_json(fault_path, {
            "schema_version": "2.0",
            "study_id": study_id,
            "task_id": task_id,
            "pair_id": pair_id,
            "task_index": task_index,
            "assignment_sha256": assignment["assignment_sha256"],
            "assignment_artifact_sha256": assignment_hash,
            **pair,
        })

        prompts = {
            "baseline": _blind_repair_prompt(payload, injected),
            "guarded": _guided_repair_prompt(
                payload,
                injected,
                pair["injected_validation"],
            ),
        }
        instructions = {
            "baseline": (
                "Repair one controlled NetOps workflow defect without "
                "deterministic validator diagnostics."
            ),
            "guarded": (
                "Repair one controlled NetOps workflow defect using the "
                "supplied deterministic validator diagnostics."
            ),
        }
        actual_order = (
            ("baseline", "guarded")
            if task_index % 2 == 1
            else ("guarded", "baseline")
        )
        order_map = {
            condition: order
            for order, condition in enumerate(actual_order, 1)
        }
        condition_results: dict[str, dict[str, Any]] = {}

        for condition in actual_order:
            prompt = prompts[condition]
            stage = (
                "blind_controlled_fault_repair"
                if condition == "baseline"
                else "validator_guided_controlled_fault_repair"
            )
            request = ModelCallRequest(
                provider=SUPPORTED_PROVIDER,
                model=str(plan["model_name"]),
                prompt=prompt,
                instructions=instructions[condition],
                temperature=plan.get("temperature"),
                max_output_tokens=int(plan["max_output_tokens"]),
                reasoning_effort=str(plan["reasoning_effort"]),
                metadata={
                    "study_id": study_id,
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "task_index": str(task_index),
                    "condition": condition,
                    "condition_order": str(order_map[condition]),
                    "stage": stage,
                    "fault_class": pair["fault_class"],
                },
            )

            repair_call: dict[str, Any] | None = None
            repaired: str | None = None
            validation_after: dict[str, Any] | None = None
            error_type: str | None = None
            error_message: str | None = None
            repair_attempts = 0
            try:
                result = hosted.call(request)
                repair_call = result.to_dict()
                repaired = result.response_text.strip()
                validation_after = validate_configuration(
                    payload,
                    repaired,
                )
                repair_attempts = int(result.attempt_count)
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                repair_attempts = 1

            model_calls_used += repair_attempts
            if model_calls_used > int(plan["maximum_model_calls"]):
                raise RuntimeError(
                    "Validator-feedback adapter exceeded frozen "
                    "model-call ceiling."
                )

            provider_path = (
                provider_dir / f"{task_id}-{condition}-repair.json"
            )
            _write_json(provider_path, {
                "schema_version": "2.0",
                "stage": stage,
                "condition": condition,
                "condition_order": order_map[condition],
                "fault_class": pair["fault_class"],
                "call": repair_call,
                "terminal_error_type": error_type,
                "terminal_error_message": error_message,
            })
            provider_relative = _relative_to_parent(
                provider_path,
                output_dir,
            )
            provider_hash = _sha256_file(provider_path)

            response_path: Path | None = None
            response_hash: str | None = None
            if repaired is not None:
                response_path = (
                    responses_dir / f"{task_id}-{condition}.txt"
                )
                response_path.write_text(
                    repaired + "\n",
                    encoding="utf-8",
                )
                response_hash = _sha256_file(response_path)

            score = (
                int(validation_after["valid"])
                if validation_after is not None
                else None
            )
            scoring_status = (
                "COMPLETED" if score in (0, 1) else "NOT_SCORED"
            )
            reason = (
                "VALID_CONFIGURATION"
                if score == 1
                else "VALIDATION_FAILED"
                if score == 0
                else "REPAIR_CALL_FAILED"
            )
            validator_trace = {
                "source_candidate": source_candidate,
                "source_validation": source_validation,
                "shared_controlled_fault_candidate": injected,
                "shared_controlled_fault_candidate_sha256": injected_hash,
                "fault_class": pair["fault_class"],
                "fault_metadata": pair["fault_metadata"],
                "injected_validation": pair["injected_validation"],
                "validator_feedback_exposed_to_model": (
                    condition == "guarded"
                ),
                "repair_call_order": order_map[condition],
                "final_configuration": repaired,
                "validation_after": validation_after,
            }
            scoring_record = {
                "schema_version": "2.0",
                "study_id": study_id,
                "episode_id": f"{task_id}-{condition}",
                "pair_id": pair_id,
                "task_id": task_id,
                "condition": condition,
                "scorer_id": VALIDATOR_ID,
                "scorer_version": VALIDATOR_VERSION,
                "scoring_rule": "full_intent_constraint_validation",
                "raw_response_sha256": response_hash,
                "score": score,
                "score_reason_code": reason,
                "scoring_status": scoring_status,
                "terminal_error_type": error_type,
                "validator_trace": validator_trace,
                "shared_source_provider_trace_path": source_provider_relative,
                "shared_source_provider_trace_sha256": source_provider_hash,
                "repair_provider_trace_path": provider_relative,
                "repair_provider_trace_sha256": provider_hash,
            }
            scoring_path = (
                scoring_dir / f"{task_id}-{condition}.json"
            )
            _write_json(scoring_path, scoring_record)

            events.append({
                "event_type": "hosted_model_call",
                "study_id": study_id,
                "task_id": task_id,
                "pair_id": pair_id,
                "condition": condition,
                "condition_order": order_map[condition],
                "stage": stage,
                "fault_class": pair["fault_class"],
                "validator_feedback_exposed_to_model": (
                    condition == "guarded"
                ),
                "outcome": (
                    "COMPLETED" if repair_call is not None else "FAILED"
                ),
                "resolved_model": (
                    repair_call.get("resolved_model")
                    if repair_call is not None
                    else None
                ),
                "model_calls_used": repair_attempts,
            })

            condition_results[condition] = {
                "prompt": prompt,
                "call": repair_call,
                "repair_attempts": repair_attempts,
                "error_type": error_type,
                "error_message": error_message,
                "response_path": response_path,
                "response_hash": response_hash,
                "score": score,
                "reason": reason,
                "scoring_status": scoring_status,
                "scoring_path": scoring_path,
                "validation_after": validation_after,
            }

        for condition in SUPPORTED_PAIRED_CONDITIONS:
            data = condition_results[condition]
            repair_call = data["call"]
            call_status = (
                "FAILED"
                if repair_call is None
                else _call_status(source_call, repair_call)
            )
            resolved_model = (
                str(repair_call.get("resolved_model"))
                if repair_call is not None
                and repair_call.get("resolved_model")
                else str(source_call.get("resolved_model"))
                if source_call is not None
                and source_call.get("resolved_model")
                else str(plan["model_version"])
            )
            scoring_path = data["scoring_path"]
            row = {
                "schema_version": "1.0",
                "result_schema_id": PAIRED_BINARY_RESULT_SCHEMA_ID,
                "study_id": study_id,
                "episode_id": f"{task_id}-{condition}",
                "pair_id": pair_id,
                "task_id": task_id,
                "task_family": TASK_FAMILY,
                "condition": condition,
                "paired_condition": (
                    "guarded" if condition == "baseline" else "baseline"
                ),
                "condition_order": order_map[condition],
                "execution_mode": str(plan["execution_mode"]),
                "model_provider": SUPPORTED_PROVIDER,
                "model_name": str(plan["model_name"]),
                "model_version": resolved_model,
                "model_configuration_sha256": model_configuration_hash,
                "task_manifest_id": "task-manifest-v2",
                "task_manifest_sha256": task_manifest_hash,
                "task_input_sha256": task_record["task_input_sha256"],
                "reference_answer_sha256": task_record[
                    "reference_answer_sha256"
                ],
                "transformation_id": plan["transformations"][condition],
                "transformation_manifest_sha256": transformation_hash,
                "prompt_sha256": _sha256_bytes(
                    data["prompt"].encode("utf-8")
                ),
                "shared_initial_candidate_path": _relative_to_parent(
                    injected_path,
                    output_dir,
                ),
                "shared_initial_candidate_sha256": injected_hash,
                "shared_initial_provider_trace_path": source_provider_relative,
                "shared_initial_provider_trace_sha256": source_provider_hash,
                "call_status": call_status,
                "attempt_count": max(
                    1,
                    source_attempts + int(data["repair_attempts"]),
                ),
                "model_calls_used": (
                    source_attempts + int(data["repair_attempts"])
                ),
                "terminal_error_type": (
                    data["error_type"] if call_status == "FAILED" else None
                ),
                "terminal_error_message": (
                    data["error_message"] if call_status == "FAILED" else None
                ),
                "response_sha256": (
                    data["response_hash"]
                    if call_status in {"COMPLETED", "CACHED"}
                    else None
                ),
                "response_artifact_path": (
                    _relative_to_parent(data["response_path"], output_dir)
                    if data["response_path"] is not None
                    and call_status in {"COMPLETED", "CACHED"}
                    else None
                ),
                "scoring_status": data["scoring_status"],
                "score": data["score"],
                "score_reason_code": data["reason"],
                "scorer_id": VALIDATOR_ID,
                "scorer_version": VALIDATOR_VERSION,
                "scoring_input_sha256": _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "shared_controlled_fault_candidate_sha256": (
                                injected_hash
                            ),
                            "fault_class": pair["fault_class"],
                            "validator_feedback_exposed_to_model": (
                                condition == "guarded"
                            ),
                            "validation_after": data["validation_after"],
                        }
                    )
                ),
                "scoring_artifact_path": _relative_to_parent(
                    scoring_path,
                    output_dir,
                ),
                "scoring_artifact_sha256": _sha256_file(scoring_path),
                "contamination_flags": [],
                "validity_flags": [],
                "started_at_utc": task_started.isoformat(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "latency_ms": max(
                    0,
                    int(
                        (
                            datetime.now(timezone.utc) - task_started
                        ).total_seconds()
                        * 1000
                    ),
                ),
            }
            row_issues = validate_paired_binary_result_row(row)
            if row_issues:
                raise RuntimeError(
                    "Generated invalid validator-feedback result row: "
                    + "; ".join(row_issues)
                )
            rows.append(row)

    results_path = output_dir / "raw_results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    execution_log_path = output_dir / "execution_log.jsonl"
    execution_log_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    artifact_paths = [
        results_path,
        result_schema_path,
        execution_log_path,
        task_manifest_path,
        transformation_path,
        model_configuration_path,
        assignment_path,
        *sorted(responses_dir.glob("*.txt")),
        *sorted(scoring_dir.glob("*.json")),
        *sorted(provider_dir.glob("*.json")),
        *sorted(fault_dir.glob("*.json")),
    ]
    call_cache_dir = output_dir / "call_cache"
    if call_cache_dir.is_dir():
        artifact_paths.extend(sorted(call_cache_dir.glob("*.json")))
    artifact_hashes = {
        _relative_to_parent(path, output_dir): _sha256_file(path)
        for path in artifact_paths
    }

    completed_count = sum(row["score"] in (0, 1) for row in rows)
    manifest = {
        "status": COMPLETED_STATUS,
        "schema_version": "1.0",
        "adapter_family": plan["adapter_family"],
        "study_id": study_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_episode_count": task_count * 2,
        "completed_episode_count": completed_count,
        "failed_episode_count": len(rows) - completed_count,
        "model_calls_used": model_calls_used,
        "maximum_model_calls": plan["maximum_model_calls"],
        "source_generation_attempt_count": task_count,
        "source_valid_count": source_valid_count,
        "source_invalid_or_failed_count": task_count - source_valid_count,
        "controlled_fault_assignment_sha256": assignment["assignment_sha256"],
        "results_path": _relative_to_parent(results_path, output_dir),
        "result_schema_path": _relative_to_parent(
            result_schema_path,
            output_dir,
        ),
        "execution_log_path": _relative_to_parent(
            execution_log_path,
            output_dir,
        ),
        "task_manifest_path": _relative_to_parent(
            task_manifest_path,
            output_dir,
        ),
        "transformation_manifest_path": _relative_to_parent(
            transformation_path,
            output_dir,
        ),
        "model_configuration_path": _relative_to_parent(
            model_configuration_path,
            output_dir,
        ),
        "result_schema_id": PAIRED_BINARY_RESULT_SCHEMA_ID,
        "result_schema_version": PAIRED_BINARY_RESULT_SCHEMA_VERSION,
        "execution_mode": str(plan["execution_mode"]),
        "artifact_hashes": artifact_hashes,
        "warnings": [],
        "preregistration_sha256": _sha256_bytes(
            _canonical_json_bytes(preregistration)
        ),
    }
    for field in (
        "master_prompt_sha256",
        "framework_commit",
        "framework_tag",
        "capability_manifest_sha256",
        "human_scientific_intervention_after_launch",
    ):
        if field in plan:
            manifest[field] = plan[field]

    _write_json(output_dir / "execution_manifest.json", manifest)
    return manifest
