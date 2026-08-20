from cnsm_agentic.autonomous_research.final_pipeline import (
    required_confirmatory_task_count,
)
from cnsm_agentic.autonomous_research.hosted_netops_adapter import (
    hosted_netops_plan_issues,
)
from cnsm_agentic.autonomous_research.repair_schemas import (
    BudgetScenario,
    ContaminationRiskPlan,
    PowerPlanningBrief,
    RepairedStudyDesign,
    TransformationValidationPlan,
)


def _design() -> RepairedStudyDesign:
    return RepairedStudyDesign(
        selected_candidate_id="study-1",
        title="Test",
        research_question="RQ",
        confirmatory_hypotheses=["H1"],
        exploratory_questions=[],
        benchmark_scope=["benchmark"],
        model_scope=["model"],
        transformation_scope=["baseline", "guarded"],
        primary_estimand="paired difference",
        secondary_estimands=[],
        sampling_plan="300 confirmatory tasks.",
        analysis_plan="Paired analysis.",
        multiplicity_plan="One confirmatory test.",
        missingness_plan="Deterministic.",
        contamination_plan=ContaminationRiskPlan(
            benchmark_names=["benchmark"],
            risk_factors=["risk"],
            detection_procedures=["procedure"],
            item_flagging_rules=["rule"],
            analysis_treatment="report",
            residual_uncertainty=["uncertainty"],
        ),
        budget_scenarios=[
            BudgetScenario(
                scenario_id="confirmatory",
                description="Confirmatory execution",
                planned_model_calls=600,
                models=["gpt-5-mini"],
                transformations=["baseline", "guarded"],
                discovery_items=0,
                confirmatory_items=300,
                task_cluster_count=10,
                feasibility_rationale="Within capability budget.",
            ),
        ],
        power_plan=PowerPlanningBrief(
            primary_estimand="paired difference",
            target_effect="effect",
            clustering_unit="task",
            calculation_method="power",
            assumptions=["assumption"],
            recommended_scenario_id="confirmatory",
            minimum_detectable_effect_notes="notes",
            sensitivity_analyses=[],
        ),
        transformation_validation=TransformationValidationPlan(
            transformation_families=["baseline", "guarded"],
            semantic_equivalence_checks=["check"],
            automatic_rejection_rules=["rule"],
            audit_sample_policy="automatic",
            mapping_validation="automatic",
            residual_risks=["risk"],
        ),
        preregistration_fields_complete=True,
        unresolved_critical_issues=[],
        remaining_noncritical_uncertainties=[],
        evidence_record_ids=["record-1"],
    )


def test_required_confirmatory_count_comes_from_recommended_scenario():
    assert required_confirmatory_task_count(_design()) == 300


def test_hosted_adapter_accepts_300_tasks_without_hidden_50_task_cap():
    task_count = 300

    plan = {
        "adapter_family": "hosted_netops_gvr_v1",
        "execution_mode": "scientific_confirmatory",
        "design": "paired_binary",
        "conditions": ["baseline", "guarded"],
        "task_families": ["intent_configuration_repair_v1"],
        "transformations": {
            "baseline": "direct_configuration_generation_v1",
            "guarded": "generate_validate_repair_v1",
        },
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "model_provider": "openai_responses",
        "model_name": "gpt-5-mini",
        "model_version": "gpt-5-mini",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "task_count": task_count,
        "task_indices": list(range(1, task_count + 1)),
        "estimated_model_calls": task_count * 2,
        "maximum_model_calls": task_count * 2,
        "reasoning_effort": "minimal",
        "maximum_attempts_per_call": 1,
        "max_output_tokens": 1000,
    }

    assert hosted_netops_plan_issues(plan) == []


def test_budget_scenario_allows_zero_discovery_items():
    scenario = BudgetScenario(
        scenario_id="confirmatory-only",
        description="Pure confirmatory execution",
        planned_model_calls=600,
        models=["gpt-5-mini"],
        transformations=["baseline", "guarded"],
        discovery_items=0,
        confirmatory_items=300,
        task_cluster_count=10,
        feasibility_rationale="No discovery phase required.",
    )

    assert scenario.discovery_items == 0
