from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    adapter_family_matches,
    validate_paired_binary_result_row,
)
from .model_providers import (
    HostedModelProvider,
    JsonFileCallCache,
    ModelCallRequest,
    OpenAIResponsesProvider,
)
from .netops_generate_validate_repair import (
    BASELINE_TRANSFORMATION,
    GUARDED_TRANSFORMATION,
    TASK_FAMILY,
    TASK_GENERATOR_ID,
    TASK_GENERATOR_VERSION,
    VALIDATOR_ID,
    VALIDATOR_VERSION,
    generate_task,
    render_reference_configuration,
    validate_configuration,
)


HOSTED_NETOPS_ADAPTER_FAMILY = "hosted_netops_gvr_v1"
HOSTED_NETOPS_ADAPTER_ALIASES = ("hosted-netops-gvr-v1",)
SUPPORTED_PROVIDER = "openai_responses"


def hosted_netops_planning_contract() -> dict[str, Any]:
    """
    Return the machine-readable execution contract that an autonomous
    planner must satisfy when selecting hosted_netops_gvr_v1.
    """
    return {
        "adapter_family": HOSTED_NETOPS_ADAPTER_FAMILY,
        "execution_mode": "scientific_confirmatory",
        "design": "paired_binary",
        "conditions": [
            "baseline",
            "guarded",
        ],
        "task_families": [
            TASK_FAMILY,
        ],
        "transformations": {
            "baseline": BASELINE_TRANSFORMATION,
            "guarded": GUARDED_TRANSFORMATION,
        },
        "generation_semantics": (
            "shared_initial_candidate"
        ),
        "independent_condition_generation": False,
        "initial_generation_calls_per_task": 1,
        "maximum_repair_calls_per_task": 1,
        "retrieval_augmented_generation": False,
        "result_schema_id": (
            PAIRED_BINARY_RESULT_SCHEMA_ID
        ),
        "result_schema_version": (
            PAIRED_BINARY_RESULT_SCHEMA_VERSION
        ),
        "model_provider": SUPPORTED_PROVIDER,
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "task_count": {
            "minimum": 1,
            "maximum": (
                "Bounded by the frozen capability manifest "
                "and model-call budget."
            ),
        },
        "episodes_per_task": 2,
        "maximum_model_calls_per_task": 2,
        "task_indices": (
            "Exactly task_count unique positive integers."
        ),
        "estimated_model_calls": (
            "Between task_count and task_count * 2: "
            "one shared initial generation per task, "
            "plus at most one guarded repair call "
            "for each task whose initial candidate "
            "fails deterministic validation."
        ),
        "maximum_model_calls": (
            "Exactly task_count * 2."
        ),
        "reasoning_effort": "minimal",
        "maximum_attempts_per_call": 1,
        "max_output_tokens": {
            "minimum": 1,
            "maximum": 2000,
        },
        "model_name": (
            "Required non-empty hosted model name."
        ),
        "model_version": (
            "Required non-empty hosted model version."
        ),
    }


def hosted_netops_plan_issues(
    plan: dict[str, Any],
    *,
    maximum_task_count: int | None = None,
) -> list[str]:
    issues: list[str] = []

    if not adapter_family_matches(
        plan,
        family=HOSTED_NETOPS_ADAPTER_FAMILY,
        aliases=HOSTED_NETOPS_ADAPTER_ALIASES,
    ):
        issues.append("Adapter family is incompatible.")

    if plan.get("execution_mode") != "scientific_confirmatory":
        issues.append(
            "Hosted adapter requires scientific_confirmatory execution mode."
        )
    if plan.get("design") != "paired_binary":
        issues.append("Hosted NetOps study requires paired_binary design.")
    if plan.get("conditions") != ["baseline", "guarded"]:
        issues.append("Conditions must be exactly baseline and guarded.")
    if plan.get("task_families") != [TASK_FAMILY]:
        issues.append("Hosted NetOps study requires the NetOps task family.")
    if plan.get("transformations") != {
        "baseline": BASELINE_TRANSFORMATION,
        "guarded": GUARDED_TRANSFORMATION,
    }:
        issues.append("Hosted NetOps transformations are incompatible.")
    if plan.get("result_schema_id") != PAIRED_BINARY_RESULT_SCHEMA_ID:
        issues.append("Result schema identifier is incompatible.")
    if (
        str(plan.get("result_schema_version"))
        != PAIRED_BINARY_RESULT_SCHEMA_VERSION
    ):
        issues.append("Result schema version is incompatible.")
    if plan.get("model_provider") != SUPPORTED_PROVIDER:
        issues.append("Hosted NetOps study requires openai_responses.")
    if not isinstance(plan.get("model_name"), str) or not str(
        plan.get("model_name")
    ).strip():
        issues.append("model_name must be non-empty.")
    if not isinstance(plan.get("model_version"), str) or not str(
        plan.get("model_version")
    ).strip():
        issues.append("model_version must be non-empty.")
    if plan.get("deterministic_automated_scoring") is not True:
        issues.append("Deterministic automated scoring is required.")
    if plan.get("requires_human_scientific_labour") is not False:
        issues.append("Human scientific labour must not be required.")

    task_count = plan.get("task_count")
    invalid_task_count = (
        not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count <= 0
    )

    if (
        not invalid_task_count
        and maximum_task_count is not None
        and task_count > maximum_task_count
    ):
        invalid_task_count = True

    if invalid_task_count:
        if maximum_task_count is None:
            issues.append(
                "task_count must be a positive integer."
            )
        else:
            issues.append(
                "task_count must be an integer from 1 to "
                f"{maximum_task_count}."
            )
    else:
        task_indices = plan.get("task_indices")
        if (
            not isinstance(task_indices, list)
            or len(task_indices) != task_count
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item <= 0
                for item in task_indices
            )
            or len(set(task_indices)) != len(task_indices)
        ):
            issues.append(
                "task_indices must contain task_count unique positive integers."
            )
        expected_calls = task_count * 2
        if plan.get("estimated_model_calls") != expected_calls:
            issues.append(
                "estimated_model_calls must equal two calls per paired task."
            )
        if plan.get("maximum_model_calls") != expected_calls:
            issues.append(
                "maximum_model_calls must equal the exact planned ceiling."
            )

    if plan.get("reasoning_effort") != "minimal":
        issues.append(
            "The hosted scientific study requires minimal reasoning effort."
        )
    if plan.get("maximum_attempts_per_call") != 1:
        issues.append(
            "The hosted scientific study requires one attempt per provider call."
        )
    max_output = plan.get("max_output_tokens")
    if (
        not isinstance(max_output, int)
        or isinstance(max_output, bool)
        or max_output <= 0
        or max_output > 2000
    ):
        issues.append("max_output_tokens must be an integer from 1 to 2000.")
    return sorted(set(issues))


def _task_prompt(task: dict[str, Any]) -> str:
    state_lines = []
    for interface, state in task["initial_state"].items():
        state_lines.append(
            f"- {interface}: admin={state['admin']}, "
            f"mtu={state['mtu']}, vlan={state['vlan']}"
        )
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Current state:\n"
        + "\n".join(state_lines)
        + "\nReturn an ordered command sequence. Command order is "
        "scientifically relevant: transient safety constraints must hold "
        "after every command, not only in the final state. Generate only "
        "commands required to satisfy the intent. Do not restate preserved "
        "settings or add no-op commands. Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>\n"
        "Do not use Markdown or explanatory prose."
    )


def _repair_prompt(
    task: dict[str, Any],
    candidate: str,
    validation: dict[str, Any],
) -> str:
    violations = "\n".join(
        f"- {item['code']}: {item['message']}"
        for item in validation["violations"]
    )
    state_lines = []
    for interface, state in task["initial_state"].items():
        state_lines.append(
            f"- {interface}: admin={state['admin']}, "
            f"mtu={state['mtu']}, vlan={state['vlan']}"
        )
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Initial state:\n"
        + "\n".join(state_lines)
        + "\nThe candidate sequence failed deterministic validation.\n"
        f"Candidate:\n{candidate}\n"
        f"Violations:\n{violations}\n"
        "Return a corrected complete ordered sequence. Ensure transient "
        "constraints hold after every command, not only in the final state. "
        "Do not include no-op or unrelated commands. Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>\n"
        "Do not use Markdown or explanatory prose."
    )


class HostedNetOpsGVRAdapter:
    family = HOSTED_NETOPS_ADAPTER_FAMILY
    aliases = HOSTED_NETOPS_ADAPTER_ALIASES
    maximum_task_count: int | None = None

    def __init__(
        self,
        provider: HostedModelProvider | None = None,
    ) -> None:
        self.provider = provider

    def planning_contract(
        self,
    ) -> dict[str, Any]:
        return hosted_netops_planning_contract()

    def compatibility_issues(
        self,
        plan: dict[str, Any],
    ) -> list[str]:
        return hosted_netops_plan_issues(
            plan,
            maximum_task_count=self.maximum_task_count,
        )

    def supports(self, plan: dict[str, Any]) -> bool:
        return not self.compatibility_issues(plan)

    def _provider_for(self, output_dir: Path, plan: dict[str, Any]):
        if self.provider is not None:
            return self.provider
        return OpenAIResponsesProvider(
            cache=JsonFileCallCache(output_dir / "call_cache"),
            maximum_attempts=int(plan["maximum_attempts_per_call"]),
            retry_backoff_seconds=float(
                plan.get("retry_backoff_seconds", 1.0)
            ),
        )

    def execute(
        self,
        *,
        plan: dict[str, Any],
        preregistration: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        issues = hosted_netops_plan_issues(
            plan,
            maximum_task_count=self.maximum_task_count,
        )
        if issues:
            raise ValueError(
                "Unsupported hosted NetOps study plan: " + "; ".join(issues)
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = output_dir / "responses"
        scoring_dir = output_dir / "scoring"
        provider_dir = output_dir / "provider_calls"
        responses_dir.mkdir(exist_ok=True)
        scoring_dir.mkdir(exist_ok=True)
        provider_dir.mkdir(exist_ok=True)

        study_id = str(plan["study_id"])
        task_count = int(plan["task_count"])
        started = datetime.now(timezone.utc).isoformat()
        provider = self._provider_for(output_dir, plan)

        tasks: list[dict[str, Any]] = []
        for index in plan["task_indices"]:
            task_id = f"task-{index:06d}"
            pair_id = f"pair-{index:06d}"
            payload = generate_task(index)
            reference = render_reference_configuration(payload)
            tasks.append({
                "schema_version": "1.0",
                "task_manifest_id": "task-manifest-v1",
                "study_id": study_id,
                "task_id": task_id,
                "pair_id": pair_id,
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
            "".join(
                json.dumps(task, sort_keys=True) + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )
        task_manifest_hash = _sha256_file(task_manifest_path)

        transformation_manifest = {
            "schema_version": "1.0",
            "study_id": study_id,
            "task_family": TASK_FAMILY,
            "conditions": {
                "baseline": {
                    "transformation_id": BASELINE_TRANSFORMATION,
                    "workflow": [
                        "shared_hosted_generation",
                        "score_shared_candidate_unchanged",
                    ],
                },
                "guarded": {
                    "transformation_id": GUARDED_TRANSFORMATION,
                    "workflow": [
                        "reuse_shared_hosted_generation",
                        "deterministic_validation",
                        "optional_hosted_repair",
                        "deterministic_revalidation",
                    ],
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
        log_events: list[dict[str, Any]] = []
        model_calls_used = 0

        for task in tasks:
            payload = task["task_payload"]
            task_started = datetime.now(timezone.utc)
            initial_call_results: list[dict[str, Any]] = []
            initial_error_type: str | None = None
            initial_error_message: str | None = None
            candidate: str | None = None
            initial_validation: dict[str, Any] | None = None

            initial_request = ModelCallRequest(
                provider=SUPPORTED_PROVIDER,
                model=str(plan["model_name"]),
                prompt=_task_prompt(payload),
                instructions=(
                    "Act as a bounded NetOps configuration generator."
                ),
                temperature=plan.get("temperature"),
                max_output_tokens=int(plan["max_output_tokens"]),
                metadata={
                    "study_id": study_id,
                    "task_id": task["task_id"],
                    "pair_id": task["pair_id"],
                    "stage": "shared_initial_generation",
                },
            )

            try:
                initial = provider.call(initial_request)
                initial_call_results.append(initial.to_dict())
                candidate = initial.response_text.strip()
                initial_validation = validate_configuration(
                    payload,
                    candidate,
                )
            except Exception as exc:
                initial_error_type = type(exc).__name__
                initial_error_message = str(exc)

            initial_calls_used = sum(
                int(item["attempt_count"])
                for item in initial_call_results
            )
            if initial_error_type is not None and initial_calls_used == 0:
                initial_calls_used = 1
            model_calls_used += initial_calls_used
            if model_calls_used > int(plan["maximum_model_calls"]):
                raise RuntimeError(
                    "Hosted pilot exceeded its frozen model-call ceiling."
                )

            shared_response_hash = None
            shared_response_relative = None
            if candidate is not None:
                shared_response_path = (
                    responses_dir
                    / f"{task['task_id']}-shared-initial.txt"
                )
                shared_response_path.write_text(
                    candidate + "\n",
                    encoding="utf-8",
                )
                shared_response_hash = _sha256_file(
                    shared_response_path
                )
                shared_response_relative = _relative_to_parent(
                    shared_response_path,
                    output_dir,
                )

            initial_provider_path = (
                provider_dir
                / f"{task['task_id']}-shared-initial.json"
            )
            _write_json(
                initial_provider_path,
                {
                    "schema_version": "1.0",
                    "study_id": study_id,
                    "task_id": task["task_id"],
                    "pair_id": task["pair_id"],
                    "stage": "shared_initial_generation",
                    "calls": initial_call_results,
                    "terminal_error_type": initial_error_type,
                    "terminal_error_message": initial_error_message,
                },
            )
            initial_provider_relative = _relative_to_parent(
                initial_provider_path,
                output_dir,
            )
            initial_provider_hash = _sha256_file(
                initial_provider_path
            )

            for item in initial_call_results:
                log_events.append(
                    {
                        "event_type": "hosted_model_call",
                        "study_id": study_id,
                        "episode_id": None,
                        "pair_id": task["pair_id"],
                        "task_id": task["task_id"],
                        "condition": "shared",
                        "stage": "shared_initial_generation",
                        "stage_index": 1,
                        "attempt_number": item["attempt_count"],
                        "cache_key_sha256": item[
                            "cache_key_sha256"
                        ],
                        "provider_request_id": item["request_id"],
                        "provider_response_id": item["response_id"],
                        "requested_model": item["requested_model"],
                        "resolved_model": item["resolved_model"],
                        "input_tokens": item["input_tokens"],
                        "output_tokens": item["output_tokens"],
                        "total_tokens": item["total_tokens"],
                        "latency_ms": item["latency_ms"],
                        "cache_status": item["cache_status"],
                        "outcome": "COMPLETED",
                        "model_calls_used": item["attempt_count"],
                    }
                )
            if not initial_call_results:
                log_events.append(
                    {
                        "event_type": "hosted_model_call",
                        "study_id": study_id,
                        "episode_id": None,
                        "pair_id": task["pair_id"],
                        "task_id": task["task_id"],
                        "condition": "shared",
                        "stage": "shared_initial_generation",
                        "outcome": "FAILED",
                        "error_type": initial_error_type,
                        "error_message": initial_error_message,
                        "model_calls_used": 0,
                    }
                )

            baseline_score = (
                int(initial_validation["valid"])
                if initial_validation is not None
                else None
            )
            guarded_score: int | None = None
            guarded_final = candidate
            guarded_validation = initial_validation
            repair_applied = False
            repair_call_results: list[dict[str, Any]] = []
            repair_provider_relative: str | None = None
            repair_provider_hash: str | None = None
            repair_error_type: str | None = None
            repair_error_message: str | None = None
            repair_prompt_hash: str | None = None
            repair_calls_used = 0

            if (
                candidate is not None
                and initial_validation is not None
                and not initial_validation["valid"]
            ):
                repair_applied = True
                repair_prompt = _repair_prompt(
                    payload,
                    candidate,
                    initial_validation,
                )
                repair_prompt_hash = _sha256_bytes(
                    repair_prompt.encode("utf-8")
                )
                repair_request = ModelCallRequest(
                    provider=SUPPORTED_PROVIDER,
                    model=str(plan["model_name"]),
                    prompt=repair_prompt,
                    instructions=(
                        "Repair the configuration using only the supplied "
                        "deterministic validation feedback."
                    ),
                    temperature=plan.get("temperature"),
                    max_output_tokens=int(plan["max_output_tokens"]),
                    reasoning_effort=str(
                        plan["reasoning_effort"]
                    ),
                    metadata={
                        "study_id": study_id,
                        "task_id": task["task_id"],
                        "pair_id": task["pair_id"],
                        "episode_id": (
                            f"{task['task_id']}-guarded"
                        ),
                        "stage": "repair",
                    },
                )
                try:
                    repaired = provider.call(repair_request)
                    repair_call_results.append(repaired.to_dict())
                    guarded_final = repaired.response_text.strip()
                    guarded_validation = validate_configuration(
                        payload,
                        guarded_final,
                    )
                except Exception as exc:
                    repair_error_type = type(exc).__name__
                    repair_error_message = str(exc)
                    guarded_final = None
                    guarded_validation = None

                repair_calls_used = sum(
                    int(item["attempt_count"])
                    for item in repair_call_results
                )
                if repair_error_type is not None and repair_calls_used == 0:
                    repair_calls_used = 1
                model_calls_used += repair_calls_used
                if model_calls_used > int(plan["maximum_model_calls"]):
                    raise RuntimeError(
                        "Hosted pilot exceeded its frozen "
                        "model-call ceiling."
                    )

                repair_provider_path = (
                    provider_dir
                    / f"{task['task_id']}-guarded-repair.json"
                )
                _write_json(
                    repair_provider_path,
                    {
                        "schema_version": "1.0",
                        "study_id": study_id,
                        "episode_id": (
                            f"{task['task_id']}-guarded"
                        ),
                        "task_id": task["task_id"],
                        "pair_id": task["pair_id"],
                        "condition": "guarded",
                        "stage": "repair",
                        "calls": repair_call_results,
                        "terminal_error_type": repair_error_type,
                        "terminal_error_message": (
                            repair_error_message
                        ),
                    },
                )
                repair_provider_relative = _relative_to_parent(
                    repair_provider_path,
                    output_dir,
                )
                repair_provider_hash = _sha256_file(
                    repair_provider_path
                )

                for item in repair_call_results:
                    log_events.append(
                        {
                            "event_type": "hosted_model_call",
                            "study_id": study_id,
                            "episode_id": (
                                f"{task['task_id']}-guarded"
                            ),
                            "pair_id": task["pair_id"],
                            "task_id": task["task_id"],
                            "condition": "guarded",
                            "stage": "repair",
                            "stage_index": 2,
                            "attempt_number": item["attempt_count"],
                            "cache_key_sha256": item[
                                "cache_key_sha256"
                            ],
                            "provider_request_id": item[
                                "request_id"
                            ],
                            "provider_response_id": item[
                                "response_id"
                            ],
                            "requested_model": item[
                                "requested_model"
                            ],
                            "resolved_model": item[
                                "resolved_model"
                            ],
                            "input_tokens": item["input_tokens"],
                            "output_tokens": item[
                                "output_tokens"
                            ],
                            "total_tokens": item["total_tokens"],
                            "latency_ms": item["latency_ms"],
                            "cache_status": item["cache_status"],
                            "outcome": "COMPLETED",
                            "model_calls_used": item[
                                "attempt_count"
                            ],
                        }
                    )
                if not repair_call_results:
                    log_events.append(
                        {
                            "event_type": "hosted_model_call",
                            "study_id": study_id,
                            "episode_id": (
                                f"{task['task_id']}-guarded"
                            ),
                            "pair_id": task["pair_id"],
                            "task_id": task["task_id"],
                            "condition": "guarded",
                            "stage": "repair",
                            "outcome": "FAILED",
                            "error_type": repair_error_type,
                            "error_message": (
                                repair_error_message
                            ),
                            "model_calls_used": 0,
                        }
                    )

            if guarded_validation is not None:
                guarded_score = int(guarded_validation["valid"])

            if baseline_score == 1 and guarded_score != 1:
                raise RuntimeError(
                    "Monotonicity violation: a valid shared baseline "
                    "candidate became invalid under the guarded workflow."
                )

            resolved_models = [
                str(item["resolved_model"])
                for item in (
                    initial_call_results + repair_call_results
                )
                if item.get("resolved_model")
            ]
            resolved_model = (
                resolved_models[-1]
                if resolved_models
                else str(plan["model_version"])
            )

            condition_payloads = {
                "baseline": {
                    "final_response": candidate,
                    "validation_after": initial_validation,
                    "score": baseline_score,
                    "repair_applied": False,
                    "repair_calls": [],
                    "repair_provider_path": None,
                    "repair_provider_hash": None,
                    "terminal_error_type": initial_error_type,
                    "terminal_error_message": (
                        initial_error_message
                    ),
                    "model_calls_used": initial_calls_used,
                },
                "guarded": {
                    "final_response": guarded_final,
                    "validation_after": guarded_validation,
                    "score": guarded_score,
                    "repair_applied": repair_applied,
                    "repair_calls": repair_call_results,
                    "repair_provider_path": (
                        repair_provider_relative
                    ),
                    "repair_provider_hash": repair_provider_hash,
                    "terminal_error_type": (
                        initial_error_type
                        or repair_error_type
                    ),
                    "terminal_error_message": (
                        initial_error_message
                        or repair_error_message
                    ),
                    "model_calls_used": (
                        initial_calls_used
                        + repair_calls_used
                    ),
                },
            }

            for order, condition in enumerate(
                SUPPORTED_PAIRED_CONDITIONS,
                1,
            ):
                episode_started = task_started
                episode_id = f"{task['task_id']}-{condition}"
                condition_data = condition_payloads[condition]
                final_response = condition_data["final_response"]
                score = condition_data["score"]
                error_type = condition_data[
                    "terminal_error_type"
                ]
                error_message = condition_data[
                    "terminal_error_message"
                ]

                response_hash = None
                response_relative = None
                normalized_response = None
                if final_response is not None:
                    response_path = (
                        responses_dir / f"{episode_id}.txt"
                    )
                    response_path.write_text(
                        final_response + "\n",
                        encoding="utf-8",
                    )
                    response_hash = _sha256_file(response_path)
                    response_relative = _relative_to_parent(
                        response_path,
                        output_dir,
                    )
                    normalized_response = final_response.strip()

                if score in (0, 1):
                    scoring_status = "COMPLETED"
                    reason = (
                        "VALID_CONFIGURATION"
                        if score
                        else "VALIDATION_FAILED"
                    )
                    call_status = (
                        "CACHED"
                        if initial_call_results
                        and all(
                            item["cache_status"] == "HIT"
                            for item in initial_call_results
                        )
                        and (
                            not condition_data["repair_calls"]
                            or all(
                                item["cache_status"] == "HIT"
                                for item in condition_data[
                                    "repair_calls"
                                ]
                            )
                        )
                        else "COMPLETED"
                    )
                else:
                    scoring_status = "NOT_SCORED"
                    reason = "CALL_FAILED"
                    call_status = "FAILED"

                validator_trace = {
                    "shared_initial_candidate": candidate,
                    "shared_initial_candidate_sha256": (
                        shared_response_hash
                    ),
                    "shared_initial_response_path": (
                        shared_response_relative
                    ),
                    "validation_before": initial_validation,
                    "repair_applied": condition_data[
                        "repair_applied"
                    ],
                    "repair_prompt_sha256": (
                        repair_prompt_hash
                        if condition == "guarded"
                        else None
                    ),
                    "final_configuration": final_response,
                    "validation_after": condition_data[
                        "validation_after"
                    ],
                }

                scoring_record = {
                    "schema_version": "1.0",
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": task["pair_id"],
                    "task_id": task["task_id"],
                    "condition": condition,
                    "scorer_id": (
                        VALIDATOR_ID
                    ),
                    "scorer_version": VALIDATOR_VERSION,
                    "scoring_rule": (
                        "full_intent_constraint_validation"
                    ),
                    "raw_response_sha256": response_hash,
                    "normalized_response": normalized_response,
                    "reference_answer_sha256": task[
                        "reference_answer_sha256"
                    ],
                    "normalized_reference_answer": task[
                        "reference_answer"
                    ],
                    "score": score,
                    "score_reason_code": reason,
                    "scoring_status": scoring_status,
                    "terminal_error_type": error_type,
                    "validator_trace": validator_trace,
                    "shared_initial_provider_trace_path": (
                        initial_provider_relative
                    ),
                    "shared_initial_provider_trace_sha256": (
                        initial_provider_hash
                    ),
                    "repair_provider_trace_path": (
                        condition_data["repair_provider_path"]
                    ),
                    "repair_provider_trace_sha256": (
                        condition_data["repair_provider_hash"]
                    ),
                }
                scoring_path = scoring_dir / f"{episode_id}.json"
                _write_json(scoring_path, scoring_record)
                scoring_relative = _relative_to_parent(
                    scoring_path,
                    output_dir,
                )
                scoring_hash = _sha256_file(scoring_path)

                completed = datetime.now(timezone.utc)
                row = {
                    "schema_version": "1.0",
                    "result_schema_id": (
                        PAIRED_BINARY_RESULT_SCHEMA_ID
                    ),
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": task["pair_id"],
                    "task_id": task["task_id"],
                    "task_family": TASK_FAMILY,
                    "condition": condition,
                    "paired_condition": (
                        "guarded"
                        if condition == "baseline"
                        else "baseline"
                    ),
                    "condition_order": order,
                    "execution_mode": str(plan["execution_mode"]),
                    "model_provider": SUPPORTED_PROVIDER,
                    "model_name": str(plan["model_name"]),
                    "model_version": resolved_model,
                    "model_configuration_sha256": (
                        model_configuration_hash
                    ),
                    "task_manifest_id": "task-manifest-v1",
                    "task_manifest_sha256": task_manifest_hash,
                    "task_input_sha256": task[
                        "task_input_sha256"
                    ],
                    "reference_answer_sha256": task[
                        "reference_answer_sha256"
                    ],
                    "transformation_id": plan[
                        "transformations"
                    ][condition],
                    "transformation_manifest_sha256": (
                        transformation_hash
                    ),
                    "prompt_sha256": _sha256_bytes(
                        _task_prompt(payload).encode("utf-8")
                    ),
                    "shared_initial_candidate_path": (
                        shared_response_relative
                    ),
                    "shared_initial_candidate_sha256": (
                        shared_response_hash
                    ),
                    "shared_initial_provider_trace_path": (
                        initial_provider_relative
                    ),
                    "shared_initial_provider_trace_sha256": (
                        initial_provider_hash
                    ),
                    "call_status": call_status,
                    "attempt_count": condition_data[
                        "model_calls_used"
                    ],
                    "model_calls_used": condition_data[
                        "model_calls_used"
                    ],
                    "terminal_error_type": error_type,
                    "terminal_error_message": error_message,
                    "response_sha256": response_hash,
                    "response_artifact_path": response_relative,
                    "scoring_status": scoring_status,
                    "score": score,
                    "score_reason_code": reason,
                    "scorer_id": (
                        VALIDATOR_ID
                    ),
                    "scorer_version": VALIDATOR_VERSION,
                    "scoring_input_sha256": _sha256_bytes(
                        _canonical_json_bytes(validator_trace)
                    ),
                    "scoring_artifact_path": scoring_relative,
                    "scoring_artifact_sha256": scoring_hash,
                    "contamination_flags": [],
                    "validity_flags": [],
                    "started_at_utc": (
                        episode_started.isoformat()
                    ),
                    "completed_at_utc": completed.isoformat(),
                    "latency_ms": max(
                        0,
                        int(
                            (
                                completed - episode_started
                            ).total_seconds()
                            * 1000
                        ),
                    ),
                }
                row_issues = validate_paired_binary_result_row(
                    row
                )
                if row_issues:
                    raise RuntimeError(
                        "Generated invalid hosted result row: "
                        + "; ".join(row_issues)
                    )
                rows.append(row)

        results_path = output_dir / "raw_results.jsonl"
        results_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        execution_log_path = output_dir / "execution_log.jsonl"
        execution_log_path.write_text(
            "".join(
                json.dumps(event, sort_keys=True) + "\n"
                for event in log_events
            ),
            encoding="utf-8",
        )

        artifact_paths = [
            results_path,
            result_schema_path,
            execution_log_path,
            task_manifest_path,
            transformation_path,
            model_configuration_path,
            *sorted(responses_dir.glob("*.txt")),
            *sorted(scoring_dir.glob("*.json")),
            *sorted(provider_dir.glob("*.json")),
        ]
        call_cache_dir = output_dir / "call_cache"
        if call_cache_dir.is_dir():
            artifact_paths.extend(sorted(call_cache_dir.glob("*.json")))
        artifact_hashes = {
            _relative_to_parent(path, output_dir): _sha256_file(path)
            for path in artifact_paths
        }

        completed_count = sum(row["score"] in (0, 1) for row in rows)
        failed_count = len(rows) - completed_count
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "status": COMPLETED_STATUS,
            "schema_version": "1.0",
            "adapter_family": self.family,
            "study_id": study_id,
            "started_at_utc": started,
            "completed_at_utc": completed_at,
            "planned_episode_count": task_count * 2,
            "completed_episode_count": completed_count,
            "failed_episode_count": failed_count,
            "model_calls_used": model_calls_used,
            "maximum_model_calls": plan["maximum_model_calls"],
            "results_path": _relative_to_parent(results_path, output_dir),
            "result_schema_path": _relative_to_parent(
                result_schema_path, output_dir
            ),
            "execution_log_path": _relative_to_parent(
                execution_log_path, output_dir
            ),
            "task_manifest_path": _relative_to_parent(
                task_manifest_path, output_dir
            ),
            "transformation_manifest_path": _relative_to_parent(
                transformation_path, output_dir
            ),
            "model_configuration_path": _relative_to_parent(
                model_configuration_path, output_dir
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
        _write_json(output_dir / "execution_manifest.json", manifest)
        return manifest
