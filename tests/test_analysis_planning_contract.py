from cnsm_agentic.autonomous_research.analysis_executors import (
    analysis_compatibility_issues,
    clear_registered_analysis_executors,
    register_builtin_analysis_executors,
    registered_analysis_planning_contracts,
)

from cnsm_agentic.autonomous_research.final_pipeline import (
    analysis_preregistration_fidelity_issues,
    preregistration_analysis_contract_issues,
)

from cnsm_agentic.autonomous_research.final_schemas import (
    PreregistrationDocument,
    PreregistrationExecutionContract,
)


def test_paired_binary_analysis_contract_is_exposed_and_compatible():
    clear_registered_analysis_executors()
    register_builtin_analysis_executors()

    contracts = registered_analysis_planning_contracts()

    assert "paired_binary_analysis_v1" in contracts

    contract = contracts["paired_binary_analysis_v1"]

    assert contract["estimand"] == (
        "paired_success_rate_difference_guarded_minus_baseline"
    )

    plan = {
        "study_id": "study-1",
        "analysis_executor": "paired_binary_analysis_v1",
        "estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "failed_call_treatment": "complete_pair_primary",
    }

    manifest = {
        "status": "COMPLETED",
        "adapter_family": "hosted_netops_gvr_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "study_id": "study-1",
        "results_path": "execution/raw_results.jsonl",
        "result_schema_path": "execution/result_schema.json",
        "artifact_hashes": {"dummy": "hash"},
        "execution_mode": "scientific_pilot",
    }

    assert analysis_compatibility_issues(
        analysis_plan=plan,
        execution_manifest=manifest,
    ) == []


def test_paired_binary_analysis_rejects_auc_composite_preregistration():
    clear_registered_analysis_executors()
    register_builtin_analysis_executors()

    contracts = registered_analysis_planning_contracts()

    preregistration = PreregistrationDocument(
        study_id="study-auc",
        title="Composite AUC study",
        research_question=(
            "Does a composite score outperform the best "
            "single submetric?"
        ),
        confirmatory_hypotheses=[
            (
                "The composite score has higher AUC-ROC "
                "than the best single submetric."
            )
        ],
        exploratory_questions=[],
        evidence_record_ids=["record-1"],
        benchmark_scope=["synthetic NetOps tasks"],
        model_scope=["gpt-5-mini"],
        transformation_scope=[
            "direct_configuration_generation_v1",
            "generate_validate_repair_v1",
        ],
        execution_contract=PreregistrationExecutionContract(
            adapter_family="hosted_netops_gvr_v1",
            execution_mode="scientific_confirmatory",
            design="paired",
            conditions=["baseline", "guarded"],
            model_provider="openai_responses",
            model_names=["gpt-5-mini"],
            task_count=200,
            planned_episode_count=400,
            maximum_model_calls=400,
            generation_semantics="shared_initial_candidate",
            independent_condition_generation=False,
            initial_generation_calls_per_task=1,
            maximum_repair_calls_per_task=1,
            retrieval_augmented_generation=False,
        ),
        primary_estimand=(
            "Difference in AUC-ROC "
            "(composite score vs best single submetric) "
            "for predicting binary simulated outage."
        ),
        primary_estimand_id=(
            "auc_composite_vs_best_submetric"
        ),
        secondary_estimands=[],
        sampling_plan="Fixed held-out paired test tasks.",
        power_and_precision_plan="Pre-specified power plan.",
        exclusion_rules=[],
        missingness_plan="No silent imputation.",
        analysis_plan=(
            "Train a composite model and compare its "
            "AUC-ROC with the best single submetric."
        ),
        multiplicity_plan="Primary estimand only.",
        contamination_plan="Check duplicate contamination.",
        stopping_rule="Complete the fixed test set.",
        planned_outputs=["analysis results"],
    )

    issues = preregistration_analysis_contract_issues(
        preregistration,
        analysis_contracts=contracts,
    )

    assert issues
    assert any(
        "not supported by any registered analysis executor"
        in issue
        for issue in issues
    )


def test_paired_binary_analysis_accepts_matching_paired_success_preregistration():
    clear_registered_analysis_executors()
    register_builtin_analysis_executors()

    contracts = registered_analysis_planning_contracts()

    preregistration = PreregistrationDocument(
        study_id="study-paired",
        title="Paired success study",
        research_question=(
            "Does the guarded condition improve success "
            "relative to baseline?"
        ),
        confirmatory_hypotheses=[
            (
                "Guarded success exceeds baseline success "
                "on paired tasks."
            )
        ],
        exploratory_questions=[],
        evidence_record_ids=["record-1"],
        benchmark_scope=["synthetic NetOps tasks"],
        model_scope=["gpt-5-mini"],
        transformation_scope=[
            "direct_configuration_generation_v1",
            "generate_validate_repair_v1",
        ],
        execution_contract=PreregistrationExecutionContract(
            adapter_family="hosted_netops_gvr_v1",
            execution_mode="scientific_confirmatory",
            design="paired",
            conditions=["baseline", "guarded"],
            model_provider="openai_responses",
            model_names=["gpt-5-mini"],
            task_count=200,
            planned_episode_count=400,
            maximum_model_calls=400,
            generation_semantics="shared_initial_candidate",
            independent_condition_generation=False,
            initial_generation_calls_per_task=1,
            maximum_repair_calls_per_task=1,
            retrieval_augmented_generation=False,
        ),
        primary_estimand=(
            "Per-task difference in verifier pass indicator "
            "(pipeline-after-up-to-1-repair) minus single-shot "
            "baseline pass indicator (paired proportion difference)."
        ),
        primary_estimand_id=(
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        secondary_estimands=[],
        sampling_plan="Fixed held-out paired test tasks.",
        power_and_precision_plan="Pre-specified precision plan.",
        exclusion_rules=[],
        missingness_plan="Complete-pair primary analysis.",
        analysis_plan=(
            "Compare guarded and baseline paired success "
            "rates using the preregistered paired analysis."
        ),
        multiplicity_plan="Primary estimand only.",
        contamination_plan="Check duplicate contamination.",
        stopping_rule="Complete the fixed test set.",
        planned_outputs=["analysis results"],
    )

    prereg_issues = preregistration_analysis_contract_issues(
        preregistration,
        analysis_contracts=contracts,
    )

    assert prereg_issues == []

    plan = {
        "study_id": "study-paired",
        "analysis_executor": "paired_binary_analysis_v1",
        "estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "failed_call_treatment": "complete_pair_primary",
    }

    issues = analysis_preregistration_fidelity_issues(
        preregistration=preregistration,
        analysis_plan=plan,
        analysis_contracts=contracts,
    )

    assert issues == []
