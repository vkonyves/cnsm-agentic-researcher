from __future__ import annotations

from typing import Any

from .execution_adapters import (
    PAIRED_BINARY_RESULT_SCHEMA_ID,
    PAIRED_BINARY_RESULT_SCHEMA_VERSION,
    adapter_family_matches,
)
from .netops_generate_validate_repair import TASK_FAMILY


HOSTED_VALIDATOR_FEEDBACK_ADAPTER_FAMILY = (
    "hosted_netops_validator_feedback_repair_v2"
)

HOSTED_VALIDATOR_FEEDBACK_ADAPTER_ALIASES = (
    "hosted-netops-validator-feedback-repair-v2",
)

BLIND_REPAIR_TRANSFORMATION = (
    "blind_controlled_fault_repair_v2"
)

GUIDED_REPAIR_TRANSFORMATION = (
    "validator_guided_controlled_fault_repair_v2"
)

SUPPORTED_PROVIDER = "openai_responses"

FAULT_PLAN_SEED = 17


def validator_feedback_planning_contract() -> dict[str, Any]:
    return {
        "adapter_family":
            HOSTED_VALIDATOR_FEEDBACK_ADAPTER_FAMILY,

        "execution_mode":
            "scientific_confirmatory",

        "design":
            "paired_binary",

        "conditions": [
            "baseline",
            "guarded",
        ],

        "task_families": [
            TASK_FAMILY,
        ],

        "transformations": {
            "baseline":
                BLIND_REPAIR_TRANSFORMATION,

            "guarded":
                GUIDED_REPAIR_TRANSFORMATION,
        },

        "generation_semantics":
            "shared_controlled_fault_candidate",

        "scientific_comparison": (
            "one-shot blind repair versus one-shot repair "
            "with deterministic validator diagnostics on "
            "the same controlled-fault candidate"
        ),

        "supports_controlled_fault_challenge":
            True,

        "controlled_fault_source":
            "deterministic_netops_fault_injector_v1",

        "fault_assignment":
            "deterministic_balanced_preoutcome",

        "fault_assignment_seed":
            FAULT_PLAN_SEED,

        "fault_assignment_is_outcome_independent":
            True,

        "shared_faulted_candidate":
            True,

        "repair_budget_matched_between_conditions":
            True,

        "guarantees_deterministic_model_sampling":
            False,

        "sampling_parameter_policy": (
            "runtime_value_recorded_in_model_configuration"
        ),

        "independent_condition_generation":
            False,

        "source_generation_calls_per_task":
            1,

        "baseline_repair_calls_per_eligible_task":
            1,

        "guarded_repair_calls_per_eligible_task":
            1,

        "retrieval_augmented_generation":
            False,

        "supports_multi_model_consensus":
            False,

        "supports_simulated_human_gate":
            False,

        "supports_prompt_family_stratification":
            False,

        "result_schema_id":
            PAIRED_BINARY_RESULT_SCHEMA_ID,

        "result_schema_version":
            PAIRED_BINARY_RESULT_SCHEMA_VERSION,

        "model_provider":
            SUPPORTED_PROVIDER,

        "deterministic_automated_scoring":
            True,

        "requires_human_scientific_labour":
            False,

        "task_count": {
            "minimum": 40,
            "multiple_of": 40,
            "maximum": (
                "Bounded by the frozen capability manifest "
                "and model-call budget."
            ),
        },

        "episodes_per_task":
            2,

        "initial_generation_calls_per_task":
            1,

        "maximum_repair_calls_per_task":
            2,

        "maximum_model_calls_per_task":
            3,

        "task_index_profile": {
            "profile_id":
                "balanced_controlled_fault_population_v2",

            "selection_is_outcome_independent":
                True,

            "task_index_rule":
                "task_indices are exactly 1..task_count",

            "combined_balance_multiple":
                40,
        },

        "task_indices": (
            "Use exactly consecutive positive integers "
            "1..task_count."
        ),

        "estimated_model_calls":
            "Exactly task_count * 3.",

        "maximum_model_calls":
            "Exactly task_count * 3.",

        "reasoning_effort":
            "minimal",

        "maximum_attempts_per_call":
            1,

        "max_output_tokens": {
            "minimum": 1,
            "maximum": 2000,
        },

        "model_name":
            "Required non-empty hosted model name.",

        "model_version":
            "Required non-empty hosted model version.",
    }


def validator_feedback_plan_issues(
    plan: dict[str, Any],
    *,
    maximum_task_count: int | None = None,
) -> list[str]:

    issues: list[str] = []

    if not adapter_family_matches(
        plan,
        family=HOSTED_VALIDATOR_FEEDBACK_ADAPTER_FAMILY,
        aliases=HOSTED_VALIDATOR_FEEDBACK_ADAPTER_ALIASES,
    ):
        issues.append(
            "Adapter family is incompatible."
        )

    if plan.get("execution_mode") != "scientific_confirmatory":
        issues.append(
            "Validator-feedback adapter requires "
            "scientific_confirmatory."
        )

    if plan.get("design") != "paired_binary":
        issues.append(
            "Validator-feedback study requires "
            "paired_binary design."
        )

    if plan.get("conditions") != [
        "baseline",
        "guarded",
    ]:
        issues.append(
            "Conditions must be exactly baseline and guarded."
        )

    if plan.get("task_families") != [
        TASK_FAMILY,
    ]:
        issues.append(
            "Validator-feedback study requires "
            "the NetOps task family."
        )

    if plan.get("transformations") != {
        "baseline":
            BLIND_REPAIR_TRANSFORMATION,
        "guarded":
            GUIDED_REPAIR_TRANSFORMATION,
    }:
        issues.append(
            "Validator-feedback transformations "
            "are incompatible."
        )

    if (
        plan.get("result_schema_id")
        != PAIRED_BINARY_RESULT_SCHEMA_ID
    ):
        issues.append(
            "Result schema identifier is incompatible."
        )

    if (
        str(plan.get("result_schema_version"))
        != PAIRED_BINARY_RESULT_SCHEMA_VERSION
    ):
        issues.append(
            "Result schema version is incompatible."
        )

    if plan.get("model_provider") != SUPPORTED_PROVIDER:
        issues.append(
            "Validator-feedback study requires "
            "openai_responses."
        )

    if (
        not isinstance(plan.get("model_name"), str)
        or not plan["model_name"].strip()
    ):
        issues.append(
            "model_name must be non-empty."
        )

    if (
        not isinstance(plan.get("model_version"), str)
        or not plan["model_version"].strip()
    ):
        issues.append(
            "model_version must be non-empty."
        )

    if (
        plan.get("deterministic_automated_scoring")
        is not True
    ):
        issues.append(
            "Deterministic automated scoring is required."
        )

    if (
        plan.get("requires_human_scientific_labour")
        is not False
    ):
        issues.append(
            "Human scientific labour must not be required."
        )

    if plan.get("initial_generation_calls_per_task") != 1:
        issues.append(
            "initial_generation_calls_per_task must be "
            "exactly 1."
        )

    if plan.get("maximum_repair_calls_per_task") != 2:
        issues.append(
            "maximum_repair_calls_per_task must be "
            "exactly 2."
        )

    if plan.get("maximum_model_calls_per_task") != 3:
        issues.append(
            "maximum_model_calls_per_task must be "
            "exactly 3."
        )

    task_count = plan.get("task_count")

    valid_count = (
        isinstance(task_count, int)
        and not isinstance(task_count, bool)
        and task_count > 0
    )

    if (
        valid_count
        and maximum_task_count is not None
        and task_count > maximum_task_count
    ):
        valid_count = False

    if not valid_count:
        issues.append(
            "task_count must be a positive integer "
            "within capability bounds."
        )

    else:
        if (
            task_count < 40
            or task_count % 40 != 0
        ):
            issues.append(
                "task_count must be at least 40 and "
                "divisible by 40 for exact fault-class "
                "and workflow-pattern balance."
            )

        expected_indices = list(
            range(1, task_count + 1)
        )

        if (
            plan.get("task_indices")
            != expected_indices
        ):
            issues.append(
                "task_indices must be exactly the "
                "consecutive range 1..task_count."
            )

        expected_calls = (
            task_count * 3
        )

        if (
            plan.get("estimated_model_calls")
            != expected_calls
        ):
            issues.append(
                "estimated_model_calls must equal "
                "task_count * 3."
            )

        if (
            plan.get("maximum_model_calls")
            != expected_calls
        ):
            issues.append(
                "maximum_model_calls must equal "
                "task_count * 3."
            )

    if plan.get("reasoning_effort") != "minimal":
        issues.append(
            "reasoning_effort must be minimal."
        )

    if plan.get("maximum_attempts_per_call") != 1:
        issues.append(
            "Exactly one attempt per provider call "
            "is required."
        )

    max_output = plan.get(
        "max_output_tokens"
    )

    if (
        not isinstance(max_output, int)
        or isinstance(max_output, bool)
        or not 1 <= max_output <= 2000
    ):
        issues.append(
            "max_output_tokens must be an integer "
            "from 1 to 2000."
        )

    return sorted(
        set(issues)
    )

class HostedNetOpsValidatorFeedbackAdapter:
    family = HOSTED_VALIDATOR_FEEDBACK_ADAPTER_FAMILY
    aliases = HOSTED_VALIDATOR_FEEDBACK_ADAPTER_ALIASES

    maximum_task_count: int | None = None


    def __init__(
        self,
        provider: Any | None = None,
    ) -> None:
        self.provider = provider
    def planning_contract(
        self,
    ) -> dict[str, Any]:
        return validator_feedback_planning_contract()

    def compatibility_issues(
        self,
        plan: dict[str, Any],
    ) -> list[str]:
        return validator_feedback_plan_issues(
            plan,
            maximum_task_count=self.maximum_task_count,
        )

    def supports(
        self,
        plan: dict[str, Any],
    ) -> bool:
        return not self.compatibility_issues(plan)

    def execute(
        self,
        *,
        plan: dict[str, Any],
        preregistration: dict[str, Any],
        output_dir,
    ) -> dict[str, Any]:
        issues = self.compatibility_issues(plan)
        if issues:
            raise ValueError(
                "Unsupported validator-feedback NetOps plan: "
                + "; ".join(issues)
            )

        from .hosted_netops_validator_feedback_execution import (
            execute_validator_feedback_study,
        )

        return execute_validator_feedback_study(
            plan=plan,
            preregistration=preregistration,
            output_dir=output_dir,
            provider=self.provider,
        )