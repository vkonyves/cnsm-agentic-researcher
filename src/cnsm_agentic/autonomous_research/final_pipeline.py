from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from agents import Agent, Runner

from .design_repair import (
    DESIGN_REPAIR_AGENT,
    READINESS_JUDGE,
    run_agent_with_retry,
)
from .evidence_verification import (
    build_evidence_alias_index,
    normalise_evidence_id,
    verify_evidence,
)
from .execution_adapters import (
    adapter_compatibility_issues,
    register_builtin_execution_adapters,
    registered_adapter_families,
    registered_adapter_planning_contracts,
    resolve_adapter,
    validate_execution_manifest,
)
from .feasibility import (
    feasibility_report,
)
from .final_agents import (
    ANALYSIS_PLANNER,
    EXPERIMENT_PLANNER,
    FINAL_JUDGE,
    MANUSCRIPT_AUTHOR,
    MANUSCRIPT_REVISER,
    PEER_REVIEWER,
    PREREGISTRATION_AGENT,
)
from .final_guardrails import (
    assert_fresh_run_dir,
    assert_no_development_inputs,
    sha256_file,
)
from .final_schemas import (
    AnalysisPlan,
    ExperimentPlan,
    FinalReadinessReport,
    ManuscriptPackage,
    PeerReviewReport,
    PreregistrationDocument,
)
from .pipeline import (
    AutonomousDiscoveryPipeline,
)
from .analysis_executors import (
    register_builtin_analysis_executors,
    analysis_compatibility_issues,
    registered_analysis_families,
    registered_analysis_planning_contracts,
    resolve_analysis_executor,
    validate_analysis_results,
)
from .repair_schemas import (
    RepairedStudyDesign,
    RepairReadinessReport,
)

from .publication_renderer import (
    build_publication_artifacts,
)


T = TypeVar("T")


def read_json(
    path: Path,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(value, "model_dump"):
        value = value.model_dump()

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_state(
    *,
    run_dir: Path,
    state: str,
    selected_candidate_id: str | None,
    development_rehearsal: bool,
    additional_fields: dict[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "state": state,
        "selected_candidate_id": (
            selected_candidate_id
        ),
        "development_rehearsal": (
            development_rehearsal
        ),
        "updated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    if additional_fields:
        value.update(
            additional_fields
        )

    write_json(
        run_dir / "state.json",
        value,
    )


async def run_agent(
    agent: Agent,
    payload: dict[str, Any],
    *,
    expected_type: type[T],
    stage_name: str,
    attempts: int = 3,
) -> T:
    """
    Run a structured-output agent with bounded retries.

    Development-run paths are prohibited from entering
    final scientific-stage prompts.
    """
    assert_no_development_inputs(
        payload
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            result = await Runner.run(
                agent,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            )

            output = result.final_output

            if not isinstance(
                output,
                expected_type,
            ):
                raise TypeError(
                    f"{stage_name} returned "
                    f"{type(output).__name__}; "
                    f"expected "
                    f"{expected_type.__name__}."
                )

            return output

        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            delay_seconds = (
                5 * (2 ** (attempt - 1))
            )

            print(
                f"{stage_name} failed on attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            print(
                "Retrying in "
                f"{delay_seconds} seconds..."
            )

            await asyncio.sleep(
                delay_seconds
            )

    raise RuntimeError(
        f"{stage_name} failed after "
        f"{attempts} attempts."
    ) from last_error


def create_failure_report(
    *,
    passed_gates: list[str],
    failed_gate: str,
    final_state: str,
    warnings: list[str] | None = None,
) -> FinalReadinessReport:
    return FinalReadinessReport(
        ready=False,
        passed_gates=passed_gates,
        failed_gates=[
            failed_gate
        ],
        warnings=warnings or [],
        final_state=final_state,
    )

def required_confirmatory_task_count(
    repaired_design: RepairedStudyDesign,
) -> int:
    """Resolve the autonomous design's selected confirmatory sample size."""
    recommended_id = (
        repaired_design.power_plan.recommended_scenario_id
    )

    matches = [
        scenario
        for scenario in repaired_design.budget_scenarios
        if scenario.scenario_id == recommended_id
    ]

    if len(matches) != 1:
        raise ValueError(
            "Exactly one budget scenario must match "
            "power_plan.recommended_scenario_id."
        )

    return int(matches[0].confirmatory_items)


def preregistration_execution_contract_issues(
    preregistration: PreregistrationDocument,
    *,
    planning_contracts: dict[str, dict[str, Any]],
    available_execution_models: list[str],
    required_task_count: int,
) -> list[str]:
    issues: list[str] = []

    declared = (
        preregistration
        .execution_contract
        .model_dump()
    )

    adapter_family = str(
        declared["adapter_family"]
    )

    adapter_contract = (
        planning_contracts.get(
            adapter_family
        )
    )

    if adapter_contract is None:
        return [
            "Preregistration selected an unregistered "
            f"adapter_family: {adapter_family}"
        ]

    for field in (
        "execution_mode",
        "design",
        "conditions",
        "model_provider",
        "generation_semantics",
        "independent_condition_generation",
        "initial_generation_calls_per_task",
        "maximum_repair_calls_per_task",
        "retrieval_augmented_generation",
    ):
        if (
            declared.get(field)
            != adapter_contract.get(field)
        ):
            issues.append(
                "Preregistration execution_contract "
                f"{field} does not match the registered "
                "adapter planning contract."
            )
        if (
            declared.get(field)
            != adapter_contract.get(field)
        ):
            issues.append(
                "Preregistration execution_contract "
                f"{field} does not match the registered "
                "adapter planning contract."
            )

    if (
        declared["model_names"]
        != available_execution_models
    ):
        issues.append(
            "Preregistration model_names must exactly "
            "match the frozen available execution models."
        )

    if (
        preregistration.model_scope
        != declared["model_names"]
    ):
        issues.append(
            "Preregistration model_scope must exactly "
            "match execution_contract.model_names."
        )

    expected_transformations = list(
        adapter_contract.get(
            "transformations",
            {},
        ).values()
    )

    if (
        preregistration.transformation_scope
        != expected_transformations
    ):
        issues.append(
            "Preregistration transformation_scope must "
            "exactly match the registered adapter "
            "transformations."
        )

    prereg_text = " ".join(
        [
            preregistration.title,
            preregistration.research_question,
            preregistration.primary_estimand,
            preregistration.sampling_plan,
            preregistration.analysis_plan,
            *preregistration.confirmatory_hypotheses,
            *preregistration.benchmark_scope,
            *preregistration.transformation_scope,
        ]
    ).lower()

    if (
        adapter_contract.get(
            "retrieval_augmented_generation"
        ) is False
        and (
            "retrieval-augmented" in prereg_text
            or "retrieval augmented" in prereg_text
            or "rag+" in prereg_text
            or "rag +" in prereg_text
        )
    ):
        issues.append(
            "Preregistration describes retrieval-augmented "
            "generation, but the registered adapter does not "
            "execute retrieval-augmented generation."
        )

    if (
        adapter_contract.get(
            "independent_condition_generation"
        ) is False
        and (
            "independent generation" in prereg_text
            or "independently generated" in prereg_text
            or "separate generation" in prereg_text
            or "separately generated" in prereg_text
        )
    ):
        issues.append(
            "Preregistration describes independent "
            "per-condition generation, but the registered "
            "adapter uses one shared initial candidate."
        )

    if (
        int(declared["task_count"])
        != required_task_count
    ):
        issues.append(
            "Preregistration task_count must equal "
            "required_confirmatory_task_count."
        )

    episodes_per_task = int(
        adapter_contract[
            "episodes_per_task"
        ]
    )

    expected_episode_count = (
        required_task_count
        * episodes_per_task
    )

    if (
        int(
            declared[
                "planned_episode_count"
            ]
        )
        != expected_episode_count
    ):
        issues.append(
            "Preregistration planned_episode_count "
            "does not match the executable adapter "
            "contract."
        )

    calls_per_task = int(
        adapter_contract[
            "maximum_model_calls_per_task"
        ]
    )

    expected_maximum_model_calls = (
        required_task_count
        * calls_per_task
    )

    if (
        int(
            declared[
                "maximum_model_calls"
            ]
        )
        != expected_maximum_model_calls
    ):
        issues.append(
            "Preregistration maximum_model_calls does "
            "not match the executable adapter contract."
        )

    return issues

def canonicalize_preregistration_execution_contract(
    preregistration: PreregistrationDocument,
    *,
    planning_contracts: dict[str, dict[str, Any]],
    available_execution_models: list[str],
    required_task_count: int,
) -> PreregistrationDocument:
    """
    Canonicalize fields that are mechanically fixed by the selected
    registered execution adapter.

    This does not alter hypotheses, estimands, analysis choices,
    benchmark scope, or other scientific content.
    """
    adapter_family = (
        preregistration.execution_contract.adapter_family
    )

    adapter_contract = planning_contracts.get(
        adapter_family
    )

    if adapter_contract is None:
        return preregistration

    episodes_per_task = int(
        adapter_contract["episodes_per_task"]
    )

    maximum_model_calls_per_task = int(
        adapter_contract[
            "maximum_model_calls_per_task"
        ]
    )

    preregistration.execution_contract.execution_mode = (
        adapter_contract["execution_mode"]
    )
    preregistration.execution_contract.design = (
        adapter_contract["design"]
    )
    preregistration.execution_contract.conditions = list(
        adapter_contract["conditions"]
    )
    preregistration.execution_contract.model_provider = (
        adapter_contract["model_provider"]
    )

    preregistration.execution_contract.model_names = list(
        available_execution_models
    )

    preregistration.execution_contract.task_count = (
        required_task_count
    )

    preregistration.execution_contract.planned_episode_count = (
        required_task_count * episodes_per_task
    )

    preregistration.execution_contract.maximum_model_calls = (
        required_task_count
        * maximum_model_calls_per_task
    )

    preregistration.execution_contract.generation_semantics = (
        adapter_contract["generation_semantics"]
    )

    preregistration.execution_contract.independent_condition_generation = (
        adapter_contract[
            "independent_condition_generation"
        ]
    )

    preregistration.execution_contract.initial_generation_calls_per_task = (
        int(
            adapter_contract[
                "initial_generation_calls_per_task"
            ]
        )
    )

    preregistration.execution_contract.maximum_repair_calls_per_task = (
        int(
            adapter_contract[
                "maximum_repair_calls_per_task"
            ]
        )
    )

    preregistration.execution_contract.retrieval_augmented_generation = (
        bool(
            adapter_contract[
                "retrieval_augmented_generation"
            ]
        )
    )

    preregistration.model_scope = list(
        available_execution_models
    )

    preregistration.transformation_scope = list(
        adapter_contract[
            "transformations"
        ].values()
    )

    return preregistration

def build_deterministic_reconciliation(
    *,
    experiment_plan: dict[str, Any],
    execution_manifest: dict[str, Any],
    analysis_results: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    confirmatory_results = (
        analysis_results.get(
            "confirmatory_results",
            [],
        )
    )

    if len(confirmatory_results) != 1:
        raise ValueError(
            "Deterministic reconciliation requires "
            "exactly one confirmatory result."
        )

    result = confirmatory_results[0]

    n_11 = int(result["n_11"])
    n_10 = int(result["n_10"])
    n_01 = int(result["n_01"])
    n_00 = int(result["n_00"])

    contingency_total = (
        n_11 + n_10 + n_01 + n_00
    )

    baseline_from_contingency = (
        n_11 + n_01
    )

    guarded_from_contingency = (
        n_11 + n_10
    )

    complete_pair_count = int(
        result["complete_pair_count"]
    )

    baseline_success_count = int(
        result["baseline_success_count"]
    )

    guarded_success_count = int(
        result["guarded_success_count"]
    )

    missingness = dict(
        analysis_results.get(
            "missingness_summary",
            {},
        )
    )

    pair_slots_from_missingness = sum(
        int(
            missingness.get(
                key,
                0,
            )
        )
        for key in (
            "complete_pairs",
            "baseline_only_observed_pairs",
            "guarded_only_observed_pairs",
            "both_missing_pairs",
        )
    )

    task_count = int(
        experiment_plan["task_count"]
    )

    conditions = list(
        experiment_plan["conditions"]
    )

    planned_episode_count = int(
        execution_manifest[
            "planned_episode_count"
        ]
    )

    completed_episode_count = int(
        execution_manifest[
            "completed_episode_count"
        ]
    )

    failed_episode_count = int(
        execution_manifest[
            "failed_episode_count"
        ]
    )

    execution_log_path = (
        run_dir
        / str(
            execution_manifest[
                "execution_log_path"
            ]
        )
    )

    cache_hit_count = 0
    cache_miss_count = 0
    hosted_model_call_event_count = 0
    completed_hosted_model_call_event_count = 0
    failed_hosted_model_call_event_count = 0

    stage_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}

    cache_key_conditions: dict[
        str,
        set[str],
    ] = {}

    if execution_log_path.is_file():
        for line in execution_log_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue

            event = json.loads(line)

            if (
                event.get("event_type")
                != "hosted_model_call"
            ):
                continue

            hosted_model_call_event_count += 1

            outcome = str(
                event.get(
                    "outcome",
                    "",
                )
            )

            if outcome == "COMPLETED":
                completed_hosted_model_call_event_count += 1
            elif outcome == "FAILED":
                failed_hosted_model_call_event_count += 1

            stage = str(
                event.get(
                    "stage",
                    "",
                )
            )

            condition = str(
                event.get(
                    "condition",
                    "",
                )
            )

            if stage:
                stage_counts[stage] = (
                    stage_counts.get(
                        stage,
                        0,
                    )
                    + 1
                )

            if condition:
                condition_counts[
                    condition
                ] = (
                    condition_counts.get(
                        condition,
                        0,
                    )
                    + 1
                )

            cache_status = event.get(
                "cache_status"
            )

            if cache_status == "HIT":
                cache_hit_count += 1
            elif cache_status == "MISS":
                cache_miss_count += 1

            cache_key = event.get(
                "cache_key_sha256"
            )

            if cache_key and condition:
                cache_key_conditions.setdefault(
                    str(cache_key),
                    set(),
                ).add(condition)

    cross_condition_cache_keys = sorted(
        cache_key
        for cache_key, seen_conditions
        in cache_key_conditions.items()
        if len(seen_conditions) > 1
    )

    marginal_contingency_consistent = (
        contingency_total
        == complete_pair_count
        and baseline_from_contingency
        == baseline_success_count
        and guarded_from_contingency
        == guarded_success_count
    )

    episode_accounting_consistent = (
        planned_episode_count
        == (
            completed_episode_count
            + failed_episode_count
        )
        and planned_episode_count
        == (
            task_count
            * len(conditions)
        )
    )

    pair_accounting_consistent = (
        pair_slots_from_missingness
        == task_count
        and int(
            missingness.get(
                "complete_pairs",
                0,
            )
        )
        == complete_pair_count
    )

    return {
        "schema_version": "1.0",
        "study_id": (
            execution_manifest[
                "study_id"
            ]
        ),
        "task_count": task_count,
        "conditions": conditions,
        "planned_episode_count": (
            planned_episode_count
        ),
        "completed_episode_count": (
            completed_episode_count
        ),
        "failed_episode_count": (
            failed_episode_count
        ),
        "complete_pair_count": (
            complete_pair_count
        ),
        "baseline_success_count": (
            baseline_success_count
        ),
        "guarded_success_count": (
            guarded_success_count
        ),
        "n_11": n_11,
        "n_10": n_10,
        "n_01": n_01,
        "n_00": n_00,
        "baseline_from_contingency": (
            baseline_from_contingency
        ),
        "guarded_from_contingency": (
            guarded_from_contingency
        ),
        "pair_total_from_contingency": (
            contingency_total
        ),
        "marginal_contingency_consistent": (
            marginal_contingency_consistent
        ),
        "pair_accounting_consistent": (
            pair_accounting_consistent
        ),
        "episode_accounting_consistent": (
            episode_accounting_consistent
        ),
        "provider_call_audit": {
            "hosted_model_call_event_count": (
                hosted_model_call_event_count
            ),
            "completed_hosted_model_call_event_count": (
                completed_hosted_model_call_event_count
            ),
            "failed_hosted_model_call_event_count": (
                failed_hosted_model_call_event_count
            ),
            "cache_hit_count": (
                cache_hit_count
            ),
            "cache_miss_count": (
                cache_miss_count
            ),
            "stage_counts": (
                stage_counts
            ),
            "condition_counts": (
                condition_counts
            ),
            "cross_condition_cache_key_reuse_observed": bool(
                cross_condition_cache_keys
            ),
            "cross_condition_cache_keys": (
                cross_condition_cache_keys
            ),
        },
        "all_deterministic_consistency_checks_passed": (
            marginal_contingency_consistent
            and pair_accounting_consistent
            and episode_accounting_consistent
            and not cross_condition_cache_keys
        ),
    }

async def create_feasible_experiment_plan(
    *,
    master_prompt: str,
    capability_manifest: dict[str, Any],
    preregistration: PreregistrationDocument,
    repaired_design: RepairedStudyDesign,
    records: list[dict[str, Any]],
    run_dir: Path,
    available_execution_models: list[str],
    maximum_attempts: int = 3,
) -> tuple[
    ExperimentPlan | None,
    dict[str, Any],
]:
    """
    Produce a capability-compliant experiment plan.

    Each generated plan is checked deterministically against the
    frozen capability manifest. Failed plans and reports are retained,
    and the planner receives the exact failures for bounded repair.
    """
    attempts_dir = (
        run_dir
        / "execution"
        / "planning_attempts"
    )
    attempts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_task_count = (
        required_confirmatory_task_count(
            repaired_design
        )
    )

    previous_plan: dict[str, Any] | None = None
    previous_issues: list[str] = []

    final_feasibility: dict[str, Any] = {
        "status": "failed",
        "issue_count": 1,
        "issues": [
            "No experiment-planning attempt completed."
        ],
    }

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        planner_payload: dict[str, Any] = {
            "master_prompt": master_prompt,
            "capability_manifest": (
                capability_manifest
            ),
            "preregistration": (
                preregistration.model_dump()
            ),
            "repaired_design": (
                repaired_design.model_dump()
            ),
            "verified_records": records,
            "available_adapter_families": (
                registered_adapter_families()
            ),
            "available_adapter_contracts": (
                registered_adapter_planning_contracts()
            ),
            "available_execution_models": (
                available_execution_models
            ),
            "required_confirmatory_task_count": (
                required_task_count
            ),
            "planning_attempt": attempt,
            "maximum_planning_attempts": (
                maximum_attempts
            ),
            "hard_execution_constraints": {
                "human_scientific_labour_allowed": (
                    capability_manifest.get(
                        "human_scientific_labour_allowed",
                        False,
                    )
                ),
                "external_partner_allowed": (
                    capability_manifest.get(
                        "external_partner_allowed",
                        False,
                    )
                ),
                "human_annotation_allowed": (
                    capability_manifest.get(
                        "human_annotation_allowed",
                        False,
                    )
                ),
                "manual_adjudication_allowed": (
                    capability_manifest.get(
                        "manual_adjudication_allowed",
                        False,
                    )
                ),
                "nda_resources_allowed": (
                    capability_manifest.get(
                        "nda_resources_allowed",
                        False,
                    )
                ),
                "private_live_lab_available": (
                    capability_manifest.get(
                        "private_live_lab_available",
                        False,
                    )
                ),
                "public_datasets_only": (
                    capability_manifest.get(
                        "public_datasets_only",
                        False,
                    )
                ),
                "autonomous_scoring_required": (
                    capability_manifest.get(
                        "autonomous_scoring_required",
                        False,
                    )
                ),
                "docker_available": (
                    capability_manifest.get(
                        "docker_available",
                        False,
                    )
                ),
                "kubernetes_available": (
                    capability_manifest.get(
                        "kubernetes_available",
                        False,
                    )
                ),
                "hosted_model_api_available": (
                    capability_manifest.get(
                        "hosted_model_api_available",
                        False,
                    )
                ),
                "cpu_execution_available": (
                    capability_manifest.get(
                        "cpu_execution_available",
                        False,
                    )
                ),
                "local_gpu": (
                    capability_manifest.get(
                        "local_gpu",
                        {
                            "available": False,
                            "memory_gb": 0,
                        },
                    )
                ),
                "maximum_planned_model_calls": (
                    capability_manifest.get(
                        "maximum_planned_model_calls"
                    )
                ),
                "maximum_wall_clock_days": (
                    capability_manifest.get(
                        "maximum_wall_clock_days"
                    )
                ),
            },
            "mandatory_planning_instruction": (
                "Treat the frozen capability manifest as a hard "
                "execution contract. Every model, execution batch, "
                "validator, scorer, transformation, fallback and "
                "dependency must be executable with the listed "
                "capabilities. Do not include unavailable resources "
                "as optional, recommended, audit, validation or "
                "future components. When no local GPU is available, "
                "use hosted model APIs or CPU-compatible methods. "
                "When human labour is prohibited, all labels, audits, "
                "scoring and validation must be autonomous. Set "
                "adapter_family to exactly one identifier from "
                "available_adapter_families. Do not invent, describe, "
                "expand, rename, or decorate the identifier. The "
                "implementation strategy, resources, batches, and "
                "result schema must fit the selected registered "
                "adapter's actual scope. "
                "The input also contains available_adapter_contracts. "
                "When selecting an adapter_family, populate every "
                "machine-readable execution-contract field in "
                "ExperimentPlan so that it satisfies that adapter's "
                "contract exactly. Do not encode required adapter "
                "contract values only in prose fields. "
                "The input also contains available_execution_models. "
                "Set model_name to exactly one identifier from that "
                "list. Do not invent, rename, qualify, or substitute "
                "a hosted model identifier. Set model_version equal "
                "to the selected model_name unless an explicitly "
                "different version is supplied by the capability "
                "contract. "
                "The input contains required_confirmatory_task_count, "
                "which is the machine-readable sample size selected by "
                "the repaired scientific design. Set task_count exactly "
                "to required_confirmatory_task_count. Do not reduce the "
                "confirmatory sample size merely to fit an adapter. If "
                "the required sample cannot fit the frozen capabilities, "
                "the plan is infeasible rather than a smaller study."
            ),
        }

        if previous_plan is not None:
            planner_payload[
                "rejected_previous_plan"
            ] = previous_plan

            planner_payload[
                "deterministic_feasibility_issues"
            ] = previous_issues

            planner_payload[
                "repair_instruction"
            ] = (
                "Repair every deterministic feasibility failure. "
                "Remove or replace the offending dependency; do not "
                "rename it or retain it as optional. Remove all local "
                "GPU, CUDA, LoRA, local 7B or 70B model, human-rater, "
                "expert-review, annotation, manual-adjudication, "
                "external-partner, NDA, private-lab and unavailable "
                "Kubernetes requirements. Preserve the scientific "
                "question and estimands where executable. Set "
                "adapter_family to exactly one identifier from "
                "available_adapter_families; never invent or decorate "
                "an adapter identifier. Preserve "
                "required_confirmatory_task_count exactly; do not repair "
                "a capability conflict by silently reducing the sealed "
                "scientific sample size."
            )

        experiment_plan = await run_agent(
            EXPERIMENT_PLANNER,
            planner_payload,
            expected_type=ExperimentPlan,
            stage_name=(
                "Experiment planning "
                f"attempt {attempt}"
            ),
        )

        plan_dict = (
            experiment_plan.model_dump()
        )

        write_json(
            attempts_dir
            / (
                "experiment_plan_attempt_"
                f"{attempt:02d}.json"
            ),
            plan_dict,
        )

        generic_feasibility = feasibility_report(
            design=plan_dict,
            capability_manifest=capability_manifest,
        )

        combined_issues = list(
            generic_feasibility.get(
                "issues",
                [],
            )
        )

        combined_issues.extend(
            adapter_compatibility_issues(
                plan_dict
            )
        )

        if plan_dict.get("study_id") != preregistration.study_id:
            combined_issues.append(
                "ExperimentPlan.study_id must equal the "
                "preregistration study_id exactly."
            )

        if plan_dict.get("task_count") != required_task_count:
            combined_issues.append(
                "task_count must equal the machine-readable "
                "required_confirmatory_task_count "
                f"({required_task_count})."
            )

        if plan_dict.get("model_name") not in available_execution_models:
            combined_issues.append(
                "model_name must be exactly one identifier from "
                "available_execution_models."
            )

        if (
            plan_dict.get("model_name")
            and plan_dict.get("model_version")
            != plan_dict.get("model_name")
        ):
            combined_issues.append(
                "model_version must equal model_name for the "
                "currently available hosted execution model."
            )

        combined_issues = sorted(
            set(combined_issues)
        )

        final_feasibility = {
            "status": (
                "passed"
                if not combined_issues
                else "failed"
            ),
            "issue_count": len(combined_issues),
            "issues": combined_issues,
        }

        write_json(
            attempts_dir
            / (
                "experiment_plan_attempt_"
                f"{attempt:02d}_feasibility.json"
            ),
            final_feasibility,
        )

        if (
            final_feasibility["status"]
            == "passed"
        ):
            return (
                experiment_plan,
                final_feasibility,
            )

        previous_plan = plan_dict
        previous_issues = list(
            final_feasibility.get(
                "issues",
                [],
            )
        )

    return (
        None,
        final_feasibility,
    )

class FinalAutonomousResearchPipeline:
    def __init__(
        self,
        *,
        model: str,
        development_rehearsal: bool,
    ) -> None:
        self.model = model
        self.development_rehearsal = (
            development_rehearsal
        )

        register_builtin_execution_adapters()
        register_builtin_analysis_executors()

        for agent in (
            DESIGN_REPAIR_AGENT,
            READINESS_JUDGE,
            PREREGISTRATION_AGENT,
            EXPERIMENT_PLANNER,
            ANALYSIS_PLANNER,
            MANUSCRIPT_AUTHOR,
            PEER_REVIEWER,
            MANUSCRIPT_REVISER,
            FINAL_JUDGE,
        ):
            agent.model = model

    async def run(
        self,
        *,
        master_prompt: str,
        run_dir: Path,
        capability_manifest: dict[str, Any],
        paper_run_constraints: dict[str, Any],
    ) -> FinalReadinessReport:
        assert_fresh_run_dir(
            run_dir,
            development_rehearsal=(
                self.development_rehearsal
            ),
        )

        assert_no_development_inputs(
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "paper_run_constraints": (
                    paper_run_constraints
                ),
            }
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for relative_directory in (
            "literature",
            "selection",
            "design",
            "preregistration",
            "execution",
            "analysis",
            "analysis/tables",
            "analysis/figures",
            "manuscript",
            "manuscript/review_rounds",
            "manuscript/revision_rounds",
            "manuscript/final",
            "disclosure",
        ):
            (
                run_dir
                / relative_directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        programme = {
            "programme_name": (
                "CNSM 2026 final autonomous run"
            ),
            "master_prompt": master_prompt,
            "topic_family": (
                "Generative AI and Large "
                "Language Models for NetOps"
            ),
            "capability_manifest": (
                capability_manifest
            ),
            "development_rehearsal": (
                self.development_rehearsal
            ),
        }

        # -------------------------------------------------
        # 1. Fresh autonomous discovery
        # -------------------------------------------------

        discovery_pipeline = (
            AutonomousDiscoveryPipeline(
                model=self.model,
                per_source_per_query=8,
                max_synthesis_records=80,
            )
        )

        decision = (
            await discovery_pipeline.run(
                programme=programme,
                run_dir=run_dir,
            )
        )

        selected_candidate_id = (
            decision.selected_candidate_id
        )

        if not selected_candidate_id:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                ],
                failed_gate=(
                    "Autonomous discovery completed without "
                    "a final selected candidate."
                ),
                final_state=(
                    "AUTONOMOUS_FINALIST_RESOLUTION_REQUIRED"
                ),
                warnings=[
                    (
                        "The discovery stage returned no "
                        "selected_candidate_id. The run was "
                        "stopped before preregistration."
                    )
                ],
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=None,
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "discovery_decision": (
                        decision.model_dump()
                        if hasattr(
                            decision,
                            "model_dump",
                        )
                        else str(decision)
                    ),
                },
            )

            return report

        literature_dir = (
            run_dir / "literature"
        )
        selection_dir = (
            run_dir / "selection"
        )
        design_dir = (
            run_dir / "design"
        )

        records = read_json(
            literature_dir
            / "records.json"
        )
        synthesis = read_json(
            literature_dir
            / "evidence_synthesis.json"
        )
        candidates = read_json(
            selection_dir
            / "candidates.json"
        )
        reviews = read_json(
            selection_dir
            / "critic_reviews.json"
        )
        decision_json = read_json(
            selection_dir
            / "decision.json"
        )
        candidate_validation = read_json(
            selection_dir
            / "candidate_validation.json"
        )

        if (
            candidate_validation.get(
                "candidate_validation_status"
            )
            != "passed"
        ):
            raise ValueError(
                "Candidate validation did not pass."
            )

        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate
            in candidates["candidates"]
        }

        if (
            selected_candidate_id
            not in candidates_by_id
        ):
            raise ValueError(
                "Selected candidate does not exist "
                "in the validated candidate set."
            )

        selected_candidate = (
            candidates_by_id[
                selected_candidate_id
            ]
        )

        # -------------------------------------------------
        # 2. Independent evidence verification
        # -------------------------------------------------

        evidence_verification = (
            verify_evidence(
                records=records,
                synthesis=synthesis,
                candidates=candidates,
                decision=decision_json,
            )
        )

        evidence_report = (
            evidence_verification.to_dict()
        )

        write_json(
            design_dir
            / "evidence_verification.json",
            evidence_report,
        )

        if (
            evidence_verification
            .critical_issues
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                ],
                failed_gate=(
                    "Evidence verification found "
                    "critical issues."
                ),
                final_state=(
                    "AUTONOMOUS_EVIDENCE_REPAIR_REQUIRED"
                ),
                warnings=(
                    evidence_verification
                    .warnings
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "evidence_critical_issues": (
                        evidence_verification
                        .critical_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 3. Autonomous design repair
        # -------------------------------------------------

        allowed_evidence_record_ids = sorted(
            {
                record["record_id"]
                for record in records
            }
        )

        repair_payload = {
            "programme": programme,
            "master_prompt": master_prompt,
            "capability_manifest": (
                capability_manifest
            ),
            "selected_candidate": (
                selected_candidate
            ),
            "selection_decision": (
                decision_json
            ),
            "critic_reviews": reviews,
            "evidence_synthesis": synthesis,
            "evidence_verification": (
                evidence_report
            ),
            "allowed_evidence_record_ids": (
                allowed_evidence_record_ids
            ),
        }

        repaired_design = (
            await run_agent_with_retry(
                DESIGN_REPAIR_AGENT,
                repair_payload,
                expected_type=(
                    RepairedStudyDesign
                ),
                stage_name=(
                    "Autonomous design repair"
                ),
            )
        )

        if (
            repaired_design
            .selected_candidate_id
            != selected_candidate_id
        ):
            raise ValueError(
                "Autonomous design repair changed "
                "the selected candidate ID."
            )

        evidence_alias_index = (
            build_evidence_alias_index(
                records
            )
        )

        unknown_repair_evidence_ids = sorted(
            evidence_id
            for evidence_id
            in repaired_design.evidence_record_ids
            if (
                normalise_evidence_id(
                    evidence_id
                )
                not in evidence_alias_index
            )
        )

        if unknown_repair_evidence_ids:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                ],
                failed_gate=(
                    "Autonomous design repair referenced "
                    "evidence that was not retrieved."
                ),
                final_state=(
                    "AUTONOMOUS_EVIDENCE_REPAIR_REQUIRED"
                ),
                warnings=[
                    (
                        "Unresolved repaired-design evidence IDs: "
                        + ", ".join(
                            unknown_repair_evidence_ids
                        )
                    )
                ],
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "unknown_repair_evidence_ids": (
                        unknown_repair_evidence_ids
                    ),
                },
            )

            return report

        write_json(
            design_dir
            / "repaired_design.json",
            repaired_design,
        )

        repair_readiness = (
            await run_agent_with_retry(
                READINESS_JUDGE,
                {
                    "programme": programme,
                    "master_prompt": (
                        master_prompt
                    ),
                    "capability_manifest": (
                        capability_manifest
                    ),
                    "selected_candidate_id": (
                        selected_candidate_id
                    ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "repaired_design": (
                        repaired_design
                        .model_dump()
                    ),
                },
                expected_type=(
                    RepairReadinessReport
                ),
                stage_name=(
                    "Design-repair readiness judgement"
                ),
            )
        )

        if (
            repair_readiness
            .selected_candidate_id
            != selected_candidate_id
        ):
            raise ValueError(
                "Design-repair readiness report "
                "candidate mismatch."
            )

        repair_ready = (
            not (
                evidence_verification
                .critical_issues
            )
            and (
                repaired_design
                .preregistration_fields_complete
            )
            and not (
                repaired_design
                .unresolved_critical_issues
            )
        )

        repair_readiness.next_state = (
            "DESIGN_REPAIRED"
            if repair_ready
            else "DESIGN_REPAIR_REQUIRED"
        )

        write_json(
            design_dir
            / "repair_readiness_report.json",
            repair_readiness,
        )

        if not repair_ready:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                ],
                failed_gate=(
                    "Autonomous design repair left "
                    "critical issues unresolved."
                ),
                final_state=(
                    "AUTONOMOUS_DESIGN_REPAIR_REQUIRED"
                ),
                warnings=(
                    repaired_design
                    .remaining_noncritical_uncertainties
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "unresolved_critical_issues": (
                        repaired_design
                        .unresolved_critical_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 4. Deterministic feasibility of repaired design
        # -------------------------------------------------

        repaired_design_feasibility = (
            feasibility_report(
                design=(
                    repaired_design
                    .model_dump()
                ),
                capability_manifest=(
                    capability_manifest
                ),
            )
        )

        write_json(
            design_dir
            / "repaired_design_feasibility.json",
            repaired_design_feasibility,
        )

        if (
            repaired_design_feasibility[
                "status"
            ]
            != "passed"
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                ],
                failed_gate=(
                    "Repaired design is infeasible "
                    "under the frozen capability manifest."
                ),
                final_state=(
                    "AUTONOMOUS_DESIGN_REPAIR_REQUIRED"
                ),
                warnings=(
                    repaired_design_feasibility[
                        "issues"
                    ]
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "feasibility_issues": (
                        repaired_design_feasibility[
                            "issues"
                        ]
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 5. Generate provisional preregistration
        # -------------------------------------------------

        available_adapter_contracts = (
            registered_adapter_planning_contracts()
        )

        available_execution_models = [
            self.model
        ]

        prereg_required_task_count = (
            required_confirmatory_task_count(
                repaired_design
            )
        )

        preregistration = None
        preregistration_contract_issues: list[str] = []

        maximum_preregistration_attempts = 3



        for prereg_attempt in range(
            1,
            maximum_preregistration_attempts + 1,
        ):
            preregistration = await run_agent(
                PREREGISTRATION_AGENT,
                {
                    "master_prompt": master_prompt,
                    "capability_manifest": (
                        capability_manifest
                    ),
                    "selected_candidate_id": (
                        selected_candidate_id
                    ),
                    "repaired_design": (
                        repaired_design.model_dump()
                    ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "verified_record_ids": (
                        allowed_evidence_record_ids
                    ),
                    "registered_adapter_planning_contracts": (
                        available_adapter_contracts
                    ),
                    "available_execution_models": (
                        available_execution_models
                    ),
                    "required_confirmatory_task_count": (
                        prereg_required_task_count
                    ),
                    "previous_contract_issues": (
                        preregistration_contract_issues
                    ),
                    "instruction": (
                        "Produce a complete provisional "
                        "preregistration that is exactly "
                        "executable under one supplied "
                        "registered adapter planning contract. "
                        "Use only the supplied available "
                        "execution models. The structured "
                        "execution_contract, model_scope, "
                        "transformation_scope, sampling plan, "
                        "and analysis plan must describe the "
                        "same experiment. Do not introduce "
                        "additional model families, conditions, "
                        "transformations, run counts, or "
                        "execution modes that are absent from "
                        "the executable contract. Set task_count "
                        "exactly to "
                        "required_confirmatory_task_count. "
                        "If previous_contract_issues is nonempty, "
                        "return a complete replacement "
                        "preregistration correcting all of them."
                    ),
                },
                expected_type=(
                    PreregistrationDocument
                ),
                stage_name=(
                    "Provisional preregistration "
                    f"attempt {prereg_attempt}"
                ),
            )

            preregistration = (
                canonicalize_preregistration_execution_contract(
                    preregistration,
                    planning_contracts=(
                        available_adapter_contracts
                    ),
                    available_execution_models=(
                        available_execution_models
                    ),
                    required_task_count=(
                        prereg_required_task_count
                    ),
                )
            )

            preregistration_contract_issues = (
                preregistration_execution_contract_issues(
                    preregistration,
                    planning_contracts=(
                        available_adapter_contracts
                    ),
                    available_execution_models=(
                        available_execution_models
                    ),
                    required_task_count=(
                        prereg_required_task_count
                    ),
                )
            )

            write_json(
                run_dir
                / "preregistration"
                / (
                    "preregistration_contract_check_"
                    f"{prereg_attempt:02d}.json"
                ),
                {
                    "attempt": prereg_attempt,
                    "passed": not (
                        preregistration_contract_issues
                    ),
                    "issues": (
                        preregistration_contract_issues
                    ),
                    "declared_execution_contract": (
                        preregistration
                        .execution_contract
                        .model_dump()
                    ),
                },
            )

            if not preregistration_contract_issues:
                break

        if (
            preregistration is None
            or preregistration_contract_issues
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                ],
                failed_gate=(
                    "Provisional preregistration could "
                    "not satisfy the executable adapter "
                    "contract after bounded autonomous repair."
                ),
                final_state=(
                    "PREREGISTRATION_EXECUTION_CONTRACT_FAILED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "preregistration_contract_issues": (
                        preregistration_contract_issues
                    ),
                },
            )

            return report

        preregistration_path = (
            run_dir
            / "preregistration"
            / "preregistration.json"
        )

        write_json(
            preregistration_path,
            preregistration,
        )

        preregistration_hash_path = (
            preregistration_path.parent
            / "preregistration.sha256"
        )

        # A preregistration is not sealed until the
        # experiment plan passes deterministic feasibility.
        if preregistration_hash_path.exists():
            preregistration_hash_path.unlink()

        # -------------------------------------------------
        # 6. Experiment planning with bounded repair
        # -------------------------------------------------

        available_adapter_families = (
            registered_adapter_families()
        )

        if not available_adapter_families:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                ],
                failed_gate=(
                    "No autonomous execution adapters "
                    "are registered."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_ADAPTER_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "available_adapter_families": [],
                },
            )

            return report

        (
            experiment_plan,
            experiment_feasibility,
        ) = await create_feasible_experiment_plan(
            master_prompt=master_prompt,
            capability_manifest=(
                capability_manifest
            ),
            preregistration=preregistration,
            repaired_design=repaired_design,
            records=records,
            run_dir=run_dir,
            available_execution_models=(
                available_execution_models
            ),
            maximum_attempts=3,
        )

        write_json(
            design_dir
            / "experiment_plan_feasibility.json",
            experiment_feasibility,
        )

        if experiment_plan is None:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                ],
                failed_gate=(
                    "The autonomous experiment planner "
                    "could not produce a capability-compliant "
                    "plan after bounded repair attempts."
                ),
                final_state=(
                    "AUTONOMOUS_EXPERIMENT_PLAN_REPAIR_REQUIRED"
                ),
                warnings=list(
                    experiment_feasibility.get(
                        "issues",
                        [],
                    )
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "experiment_plan_repair_attempts": 3,
                    "feasibility_issues": list(
                        experiment_feasibility.get(
                            "issues",
                            [],
                        )
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 7. Accept plan and seal preregistration
        # -------------------------------------------------

        experiment_plan_path = (
            run_dir
            / "execution"
            / "experiment_plan.json"
        )

        write_json(
            experiment_plan_path,
            experiment_plan,
        )

        preregistration_hash = (
            sha256_file(
                preregistration_path
            )
        )

        preregistration_hash_path.write_text(
            preregistration_hash + "\n",
            encoding="utf-8",
        )

        write_json(
            run_dir
            / "preregistration"
            / "sealing_manifest.json",
            {
                "preregistration_sha256": (
                    preregistration_hash
                ),
                "selected_candidate_id": (
                    selected_candidate_id
                ),
                "repaired_design_feasibility": (
                    "passed"
                ),
                "experiment_plan_feasibility": (
                    "passed"
                ),
                "experiment_plan_path": str(
                    experiment_plan_path.relative_to(
                        run_dir
                    )
                ),
                "sealed_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
        )

        # -------------------------------------------------
        # 8. Resolve autonomous execution adapter
        # -------------------------------------------------

        adapter = resolve_adapter(
            experiment_plan.model_dump()
        )

        if adapter is None:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
                ],
                failed_gate=(
                    "No installed autonomous execution "
                    "adapter supports the repaired study."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_ADAPTER_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
            )

            return report

        # -------------------------------------------------
        # 9. Autonomous experiment execution
        # -------------------------------------------------

        execution_manifest = adapter.execute(
            plan=experiment_plan.model_dump(),
            preregistration=(
                preregistration.model_dump()
            ),
            output_dir=(
                run_dir / "execution"
            ),
        )

        write_json(
            run_dir
            / "execution"
            / "execution_manifest.json",
            execution_manifest,
        )

        execution_manifest_issues = (
            validate_execution_manifest(
                execution_manifest,
                plan=(
                    experiment_plan.model_dump()
                ),
                output_dir=(
                    run_dir / "execution"
                ),
                maximum_model_calls=(
                    capability_manifest.get(
                        "maximum_planned_model_calls"
                    )
                ),
            )
        )

        if execution_manifest_issues:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
                    "execution_adapter_resolved",
                ],
                failed_gate=(
                    "Autonomous execution adapter "
                    "did not produce a valid completed "
                    "execution manifest."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_INCOMPLETE"
                ),
                warnings=(
                    execution_manifest_issues
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "execution_manifest_issues": (
                        execution_manifest_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 10. Analysis planning and execution
        # -------------------------------------------------

        available_analysis_families = (
            registered_analysis_families()
        )

        if not available_analysis_families:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
                    "execution_adapter_resolved",
                    "execution_completed",
                ],
                failed_gate=(
                    "No deterministic analysis executors "
                    "are registered."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_EXECUTOR_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "available_analysis_families": [],
                },
            )

            return report

        analysis_attempts_dir = (
            run_dir
            / "analysis"
            / "planning_attempts"
        )
        analysis_attempts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        available_analysis_contracts = (
            registered_analysis_planning_contracts()
        )

        previous_analysis_plan: dict[str, Any] | None = None
        previous_analysis_issues: list[str] = []
        analysis_plan: AnalysisPlan | None = None

        for attempt in range(1, 4):
            analysis_payload: dict[str, Any] = {
                "master_prompt": master_prompt,
                "capability_manifest": capability_manifest,
                "preregistration": preregistration.model_dump(),
                "experiment_plan": experiment_plan.model_dump(),
                "execution_manifest": execution_manifest,
                "available_analysis_families": (
                    available_analysis_families
                ),
                "available_analysis_contracts": (
                    available_analysis_contracts
                ),
            }

            if previous_analysis_plan is not None:
                analysis_payload[
                    "rejected_previous_analysis_plan"
                ] = previous_analysis_plan
                analysis_payload[
                    "deterministic_analysis_issues"
                ] = previous_analysis_issues
                analysis_payload[
                    "repair_instruction"
                ] = (
                    "Repair every deterministic analysis compatibility "
                    "failure while preserving the sealed preregistration. "
                    "Use exact machine-readable identifiers from the "
                    "selected available_analysis_contracts entry. Do not "
                    "paraphrase estimand or failed-call-treatment "
                    "identifiers."
                )

            candidate_analysis_plan = await run_agent(
                ANALYSIS_PLANNER,
                analysis_payload,
                expected_type=AnalysisPlan,
                stage_name=(
                    "Analysis planning "
                    f"attempt {attempt}"
                ),
            )

            candidate_dict = (
                candidate_analysis_plan.model_dump()
            )

            write_json(
                analysis_attempts_dir
                / (
                    "analysis_plan_attempt_"
                    f"{attempt:02d}.json"
                ),
                candidate_dict,
            )

            issues = analysis_compatibility_issues(
                analysis_plan=candidate_dict,
                execution_manifest=execution_manifest,
            )

            write_json(
                analysis_attempts_dir
                / (
                    "analysis_plan_attempt_"
                    f"{attempt:02d}_compatibility.json"
                ),
                {
                    "compatible": not issues,
                    "issues": issues,
                },
            )

            if not issues:
                analysis_plan = candidate_analysis_plan
                break

            previous_analysis_plan = candidate_dict
            previous_analysis_issues = list(issues)

        if analysis_plan is None:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                ],
                failed_gate=(
                    "The autonomous analysis planner could not "
                    "produce an executor-compatible analysis plan "
                    "after bounded repair attempts."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_PLAN_REPAIR_REQUIRED"
                ),
                warnings=previous_analysis_issues,
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "analysis_compatibility_issues": (
                        previous_analysis_issues
                    ),
                },
            )

            return report

        write_json(
            run_dir
            / "analysis"
            / "analysis_plan.json",
            analysis_plan,
        )

        analysis_executor = (
            resolve_analysis_executor(
                analysis_plan=(
                    analysis_plan.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
            )
        )

        if analysis_executor is None:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                    "analysis_plan",
                ],
                failed_gate=(
                    "No installed deterministic "
                    "analysis executor supports the "
                    "analysis plan and execution output."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_EXECUTOR_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
            )

            return report

        analysis_results = (
            analysis_executor.execute(
                analysis_plan=(
                    analysis_plan.model_dump()
                ),
                preregistration=(
                    preregistration.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
                run_dir=run_dir,
            )
        )

        results_path = (
            run_dir
            / "analysis"
            / "results.json"
        )

        write_json(
            results_path,
            analysis_results,
        )

        analysis_result_issues = (
            validate_analysis_results(
                analysis_results,
                run_dir=run_dir,
                execution_manifest=(
                    execution_manifest
                ),
            )
        )

        if analysis_result_issues:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                    "analysis_plan",
                    "analysis_executor_resolved",
                ],
                failed_gate=(
                    "Deterministic analysis executor "
                    "did not produce valid completed "
                    "analysis results."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_INCOMPLETE"
                ),
                warnings=(
                    analysis_result_issues
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "analysis_result_issues": (
                        analysis_result_issues
                    ),
                },
            )

            return report

        deterministic_reconciliation = (
            build_deterministic_reconciliation(
                experiment_plan=(
                    experiment_plan.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
                analysis_results=(
                    analysis_results
                ),
                run_dir=run_dir,
            )
        )

        write_json(
            run_dir
            / "analysis"
            / "deterministic_reconciliation.json",
            deterministic_reconciliation,
        )

        # -------------------------------------------------
        # 11. Autonomous manuscript generation
        # -------------------------------------------------

        draft = await run_agent(
            MANUSCRIPT_AUTHOR,
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "verified_records": records,
                "evidence_verification": (
                    evidence_report
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_plan": (
                    analysis_plan.model_dump()
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
            },
            expected_type=ManuscriptPackage,
            stage_name=(
                "Manuscript drafting"
            ),
        )

        write_json(
            run_dir
            / "manuscript"
            / "draft_package.json",
            draft,
        )

        # -------------------------------------------------
        # 12-13. Bounded autonomous peer review / revision
        # -------------------------------------------------

        review_rounds_dir = (
            run_dir
            / "manuscript"
            / "review_rounds"
        )
        review_rounds_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        revision_rounds_dir = (
            run_dir
            / "manuscript"
            / "revision_rounds"
        )
        revision_rounds_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_manuscript = draft
        latest_peer_review: PeerReviewReport | None = None

        maximum_peer_review_rounds = 3

        for review_round in range(
            1,
            maximum_peer_review_rounds + 1,
        ):
            latest_peer_review = await run_agent(
                PEER_REVIEWER,
                {
                    "master_prompt": master_prompt,
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                        execution_manifest
                    ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript": (
                        current_manuscript.model_dump()
                    ),
                    "review_round": review_round,
                },
                expected_type=PeerReviewReport,
                stage_name=(
                    "AI peer review "
                    f"round {review_round}"
                ),
            )

            write_json(
                review_rounds_dir
                / f"review_{review_round:02d}.json",
                latest_peer_review,
            )

            if (
                latest_peer_review
                .accept_for_finalisation
            ):
                break

            if (
                review_round
                >= maximum_peer_review_rounds
            ):
                break

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "master_prompt": master_prompt,
                    "verified_records": records,
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                        execution_manifest
                    ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript": (
                        current_manuscript
                        .model_dump()
                    ),
                    "peer_review": (
                        latest_peer_review
                        .model_dump()
                    ),
                    "revision_round": (
                        review_round
                    ),
                    "revision_instruction": (
                        "Address every required revision that can be "
                        "resolved from existing verified evidence, "
                        "execution artifacts, and analysis results. "
                        "Do not invent new experiments, data, references, "
                        "statistics, URLs, artifact locations, or empirical "
                        "claims. If a reviewer request cannot be resolved "
                        "from existing artifacts, preserve it explicitly "
                        "as an unresolved limitation rather than fabricating "
                        "support."
                        " Treat deterministic_reconciliation "
                        "as authoritative for arithmetic "
                        "consistency, execution accounting, "
                        "and observed cache-key reuse. Do not "
                        "adopt a reviewer claim that contradicts "
                        "a passing deterministic reconciliation "
                        "check. If reviewer prose conflicts with "
                        "the deterministic artifact, preserve the "
                        "artifact-backed facts and explicitly "
                        "resolve the reviewer concern from those "
                        "facts."
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Autonomous manuscript revision "
                    f"round {review_round}"
                ),
            )

            write_json(
                revision_rounds_dir
                / (
                    "revised_package_"
                    f"{review_round:02d}.json"
                ),
                revised_manuscript,
            )

            current_manuscript = (
                revised_manuscript
            )

        if latest_peer_review is None:
            raise RuntimeError(
                "Autonomous peer review produced "
                "no review artifact."
            )

        revised_manuscript = (
            current_manuscript
        )

        write_json(
            run_dir
            / "manuscript"
            / "revised_package.json",
            revised_manuscript,
        )

        # -------------------------------------------------
        # 14. Deterministic IEEE rendering and page validation
        # -------------------------------------------------

        publication_dir = (
            run_dir
            / "manuscript"
            / "final"
        )

        maximum_format_revision_rounds = 2
        publication_validation: dict[str, Any] | None = None

        for format_round in range(
            0,
            maximum_format_revision_rounds + 1,
        ):
            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                publication_dir
                / (
                    "publication_validation_"
                    f"{format_round:02d}.json"
                ),
                publication_validation,
            )

            if publication_validation.get(
                "passed"
            ):
                break

            compile_status = (
                publication_validation.get(
                    "compile_status"
                )
            )

            # Manuscript revision is appropriate only for a
            # successfully compiled paper that exceeds the
            # frozen page limit. Compilation failures are
            # infrastructure/rendering failures, not reasons
            # to alter scientific manuscript content.
            if compile_status != "passed":
                break

            if publication_validation.get(
                "within_page_limit"
            ):
                break

            if (
                format_round
                >= maximum_format_revision_rounds
            ):
                break

            format_feedback = {
                "page_count": (
                    publication_validation.get(
                        "page_count"
                    )
                ),
                "maximum_pages": (
                    publication_validation.get(
                        "maximum_pages"
                    )
                ),
                "within_page_limit": (
                    publication_validation.get(
                        "within_page_limit"
                    )
                ),
                "references_included_in_limit": (
                    publication_validation.get(
                        "references_included_in_limit"
                    )
                ),
                "disclosure_statement_included_in_limit": (
                    publication_validation.get(
                        "disclosure_statement_included_in_limit"
                    )
                ),
                "template_manipulation_prohibited": (
                    publication_validation.get(
                        "template_manipulation_prohibited"
                    )
                ),
            }

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "master_prompt": master_prompt,
                    "verified_records": records,
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                        execution_manifest
                    ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript": (
                        revised_manuscript
                        .model_dump()
                    ),
                    "peer_review": (
                        latest_peer_review
                        .model_dump()
                    ),
                    "publication_validation": (
                        format_feedback
                    ),
                    "revision_round": (
                        "format_"
                        f"{format_round + 1}"
                    ),
                    "revision_instruction": (
                        "The manuscript compiled successfully but exceeds "
                        "the frozen IEEE page limit. Shorten and compact "
                        "the manuscript sufficiently to satisfy the page "
                        "limit while preserving supported scientific "
                        "claims, the primary methods and results, reviewer-"
                        "resolved information, required references, and the "
                        "mandatory Disclosure Statement. Do not change "
                        "empirical results, add unsupported claims, remove "
                        "required disclosure content, manipulate the IEEE "
                        "template, shrink fonts or margins, or invent new "
                        "evidence."
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Autonomous manuscript format revision "
                    f"{format_round + 1}"
                ),
            )

            write_json(
                revision_rounds_dir
                / (
                    "format_revised_package_"
                    f"{format_round + 1:02d}.json"
                ),
                revised_manuscript,
            )

            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

        if publication_validation is None:
            raise RuntimeError(
                "Publication validation did not execute."
            )

        write_json(
            publication_dir
            / "publication_validation.json",
            publication_validation,
        )

        # -------------------------------------------------
        # 15. Final autonomous readiness judgement
        # -------------------------------------------------

        final_report = await run_agent(
            FINAL_JUDGE,
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "paper_run_constraints": (
                    paper_run_constraints
                ),
                "publication_validation": (
                    publication_validation
                ),
                "evidence_verification": (
                    evidence_report
                ),
                "repaired_design": (
                    repaired_design.model_dump()
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
                "manuscript": (
                    revised_manuscript
                    .model_dump()
                ),
                "peer_review": (
                    latest_peer_review
                    .model_dump()
                ),
            },
            expected_type=(
                FinalReadinessReport
            ),
            stage_name=(
                "Final readiness judgement"
            ),
        )

        write_json(
            run_dir
            / "final_readiness_report.json",
            final_report,
        )

        write_state(
            run_dir=run_dir,
            state=final_report.final_state,
            selected_candidate_id=(
                selected_candidate_id
            ),
            development_rehearsal=(
                self.development_rehearsal
            ),
            additional_fields={
                "ready": final_report.ready,
            },
        )

        return final_report