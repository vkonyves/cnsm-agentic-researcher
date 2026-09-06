from cnsm_agentic.autonomous_research.final_pipeline import (
    repaired_design_adapter_capability_issues,
)


NO_RAG_ADAPTERS = {
    "hosted_netops_gvr_v1": {
        "retrieval_augmented_generation": False,
        "independent_condition_generation": False,
        "supports_multi_model_consensus": False,
        "supports_simulated_human_gate": False,
        "supports_prompt_family_stratification": False,
    }
}


def _design(**overrides):
    design = {
        "research_question": (
            "Does deterministic validation plus bounded repair reduce "
            "actionable network misconfigurations?"
        ),
        "confirmatory_hypotheses": [
            "Guarded validation and repair reduce paired failure rate."
        ],
        "primary_estimand": (
            "Paired success-rate difference guarded minus baseline."
        ),
        "secondary_estimands": [],
        "sampling_plan": (
            "Use the frozen challenging-workflow stress-test population."
        ),
        "analysis_plan": (
            "Use paired binary analysis and exact McNemar testing."
        ),
        "transformation_scope": [],
    }
    design.update(overrides)
    return design


def test_rag_dependent_science_is_rejected_when_adapter_disables_rag():
    issues = repaired_design_adapter_capability_issues(
        _design(
            research_question=(
                "Does poisoning retrieved RAG context increase "
                "actionable network misconfiguration?"
            ),
            confirmatory_hypotheses=[
                "RAG poisoning increases actionable misconfiguration."
            ],
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
    )

    assert any(
        "retrieval_augmented_generation" in issue
        for issue in issues
    )


def test_explicitly_disabled_rag_does_not_create_false_requirement():
    issues = repaired_design_adapter_capability_issues(
        _design(
            research_question=(
                "Does deterministic validation reduce failures when "
                "retrieval-augmented generation is disabled?"
            ),
            analysis_plan=(
                "No RAG is used; analyze paired validator/repair outcomes."
            ),
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
    )

    assert not any(
        "retrieval_augmented_generation" in issue
        for issue in issues
    )


def test_executable_validator_repair_design_passes_adapter_science_gate():
    issues = repaired_design_adapter_capability_issues(
        _design(),
        available_adapter_contracts=NO_RAG_ADAPTERS,
    )

    assert issues == []


def test_independent_arm_generation_is_rejected_when_unavailable():
    issues = repaired_design_adapter_capability_issues(
        _design(
            sampling_plan=(
                "Use independent per-condition sampling for baseline "
                "and guarded arms."
            ),
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
    )

    assert any(
        "independent_condition_generation" in issue
        for issue in issues
    )


def test_repaired_design_rejects_model_outside_frozen_model_set():
    issues = repaired_design_adapter_capability_issues(
        _design(
            model_scope=[
                "Primary model: openai/gpt-4o-mini@2026-08-01"
            ],
            budget_scenarios=[
                {
                    "scenario_id": "recommended",
                    "description": "Primary execution.",
                    "planned_model_calls": 120,
                    "models": [
                        "openai/gpt-4o-mini@2026-08-01"
                    ],
                    "transformations": [],
                    "discovery_items": 40,
                    "confirmatory_items": 40,
                    "task_cluster_count": 40,
                    "feasibility_rationale": "Fits adapter.",
                }
            ],
            power_plan={
                "recommended_scenario_id": "recommended",
            },
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
        available_execution_models=["gpt-5-mini"],
        maximum_planned_model_calls=120,
    )

    assert any(
        "gpt-4o-mini" in issue
        and "frozen execution-model" in issue
        for issue in issues
    )


def test_provider_prefix_and_version_of_frozen_model_are_allowed():
    issues = repaired_design_adapter_capability_issues(
        _design(
            model_scope=[
                "Primary model: openai/gpt-5-mini@2026-08-01"
            ],
            budget_scenarios=[
                {
                    "scenario_id": "recommended",
                    "description": "Primary execution.",
                    "planned_model_calls": 120,
                    "models": [
                        "openai/gpt-5-mini@2026-08-01"
                    ],
                    "transformations": [],
                    "discovery_items": 40,
                    "confirmatory_items": 40,
                    "task_cluster_count": 40,
                    "feasibility_rationale": "Fits adapter.",
                }
            ],
            power_plan={
                "recommended_scenario_id": "recommended",
            },
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
        available_execution_models=["gpt-5-mini"],
        maximum_planned_model_calls=120,
    )

    assert not any(
        "frozen execution-model" in issue
        for issue in issues
    )


def test_budget_scenario_above_frozen_call_limit_is_rejected():
    issues = repaired_design_adapter_capability_issues(
        _design(
            budget_scenarios=[
                {
                    "scenario_id": "too_large",
                    "description": "Primary execution.",
                    "planned_model_calls": 121,
                    "models": ["gpt-5-mini"],
                    "transformations": [],
                    "discovery_items": 40,
                    "confirmatory_items": 40,
                    "task_cluster_count": 40,
                    "feasibility_rationale": "Candidate scenario.",
                }
            ],
            power_plan={
                "recommended_scenario_id": "too_large",
            },
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
        available_execution_models=["gpt-5-mini"],
        maximum_planned_model_calls=120,
    )

    assert any(
        "121 > 120" in issue
        for issue in issues
    )


def test_additional_executable_calibration_is_rejected():
    issues = repaired_design_adapter_capability_issues(
        _design(
            sampling_plan=(
                "Execute 40 confirmatory tasks. "
                "A disjoint calibration run will be executed "
                "on indices 41 through 80."
            ),
            budget_scenarios=[
                {
                    "scenario_id": "recommended",
                    "description": (
                        "Confirmatory execution uses 120 calls; "
                        "calibration model calls are additional "
                        "to confirmatory execution."
                    ),
                    "planned_model_calls": 120,
                    "models": ["gpt-5-mini"],
                    "transformations": [],
                    "discovery_items": 40,
                    "confirmatory_items": 40,
                    "task_cluster_count": 40,
                    "feasibility_rationale": "Adapter geometry.",
                }
            ],
            power_plan={
                "recommended_scenario_id": "recommended",
            },
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
        available_execution_models=["gpt-5-mini"],
        maximum_planned_model_calls=120,
    )

    assert any(
        "additional calibration/pilot/alternate-model execution"
        in issue
        for issue in issues
    )

    assert any(
        "full frozen model-call budget"
        in issue
        for issue in issues
    )


def test_future_calibration_discussion_is_allowed():
    issues = repaired_design_adapter_capability_issues(
        _design(
            sampling_plan=(
                "Execute only the frozen confirmatory task set. "
                "A separate calibration study could be useful "
                "in future work but is not executed in this study."
            ),
            budget_scenarios=[
                {
                    "scenario_id": "recommended",
                    "description": "Confirmatory execution only.",
                    "planned_model_calls": 120,
                    "models": ["gpt-5-mini"],
                    "transformations": [],
                    "discovery_items": 40,
                    "confirmatory_items": 40,
                    "task_cluster_count": 40,
                    "feasibility_rationale": "Adapter geometry.",
                }
            ],
            power_plan={
                "recommended_scenario_id": "recommended",
            },
        ),
        available_adapter_contracts=NO_RAG_ADAPTERS,
        available_execution_models=["gpt-5-mini"],
        maximum_planned_model_calls=120,
    )

    assert not any(
        "additional calibration/pilot/alternate-model execution"
        in issue
        for issue in issues
    )
