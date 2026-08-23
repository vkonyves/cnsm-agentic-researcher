import pytest
from pydantic import ValidationError

from cnsm_agentic.autonomous_research.final_schemas import (
    AnalysisPlan,
    ExperimentPlan,
    ExperimentTransformations,
    PreregistrationDocument,
    PreregistrationExecutionContract,
)


def test_zero_calls_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentPlan(
            study_id="s",
            adapter_family="hosted_netops_gvr_v1",
            execution_mode="scientific_confirmatory",
            design="paired_binary",
            conditions=["baseline", "guarded"],
            task_families=[
                "intent_configuration_repair_v1"
            ],
            transformations=ExperimentTransformations(
                baseline=(
                    "direct_configuration_generation_v1"
                ),
                guarded=(
                    "generate_validate_repair_v1"
                ),
            ),
            result_schema_id="paired_binary_episode_v1",
            result_schema_version="1.0",
            model_provider="openai_responses",
            model_name="gpt-5-mini",
            model_version="gpt-5-mini",
            deterministic_automated_scoring=True,
            requires_human_scientific_labour=False,
            task_count=1,
            task_indices=[1],
            estimated_model_calls=0,
            maximum_model_calls=2,
            reasoning_effort="minimal",
            maximum_attempts_per_call=1,
            max_output_tokens=1000,
            implementation_strategy="i",
            public_resources=["r"],
            model_and_version_plan=["gpt-5-mini"],
            task_manifest_strategy="t",
            transformation_manifest_strategy="x",
            execution_batches=["b"],
            randomisation_plan="r",
            caching_plan="c",
            failure_recovery_plan="f",
            result_schema="s",
            estimated_compute_notes="n",
        )


def test_unresolved_prereg_rejected() -> None:
    with pytest.raises(ValueError):
        PreregistrationDocument(
            study_id="s",
            title="t",
            research_question="q",
            confirmatory_hypotheses=["h"],
            exploratory_questions=[],
            evidence_record_ids=["r"],
            benchmark_scope=["b"],
            model_scope=["gpt-5-mini"],
            transformation_scope=[
                "direct_configuration_generation_v1",
                "generate_validate_repair_v1",
            ],
            execution_contract=(
                PreregistrationExecutionContract(
                    adapter_family=(
                        "hosted_netops_gvr_v1"
                    ),
                    execution_mode=(
                        "scientific_confirmatory"
                    ),
                    design="paired_binary",
                    conditions=[
                        "baseline",
                        "guarded",
                    ],
                    model_provider="openai_responses",
                    model_names=["gpt-5-mini"],
                    task_count=10,
                    planned_episode_count=20,
                    maximum_model_calls=20,
                )
            ),
            primary_estimand="e",
            secondary_estimands=[],
            sampling_plan="s",
            power_and_precision_plan="p",
            exclusion_rules=[],
            missingness_plan="m",
            analysis_plan="a",
            multiplicity_plan="m",
            contamination_plan="c",
            stopping_rule="s",
            planned_outputs=["o"],
            unresolved_critical_issues=["x"],
        )


def test_analysis_plan_accepts_registered_executor_field() -> None:
    plan = AnalysisPlan(
        study_id="study-1",
        analysis_executor="paired_binary_analysis_v1",
        estimand=(
            "paired_success_rate_difference_"
            "guarded_minus_baseline"
        ),
        primary_analysis=(
            "Paired difference in binary success rates."
        ),
        secondary_analyses=[],
        sensitivity_analyses=[],
        uncertainty_quantification=(
            "Paired bootstrap confidence interval."
        ),
        multiplicity_implementation=(
            "No multiplicity adjustment for one primary "
            "estimand."
        ),
        contamination_analysis=(
            "Report deterministic contamination flags."
        ),
        failed_call_treatment=(
            "Report failures and apply the "
            "preregistered rule."
        ),
        table_specifications=[],
        figure_specifications=[],
    )

    assert (
        plan.analysis_executor
        == "paired_binary_analysis_v1"
    )


def test_analysis_plan_requires_executor_field() -> None:
    with pytest.raises(ValueError):
        AnalysisPlan(
            study_id="study-1",
            estimand=(
                "paired_success_rate_difference_"
                "guarded_minus_baseline"
            ),
            primary_analysis="Primary analysis.",
            secondary_analyses=[],
            sensitivity_analyses=[],
            uncertainty_quantification=(
                "Confidence interval."
            ),
            multiplicity_implementation="None.",
            contamination_analysis="None.",
            failed_call_treatment="Report failures.",
            table_specifications=[],
            figure_specifications=[],
        )


def test_preregistration_execution_contract_checks_episode_count() -> None:
    with pytest.raises(ValidationError):
        PreregistrationExecutionContract(
            adapter_family="hosted_netops_gvr_v1",
            execution_mode="scientific_confirmatory",
            design="paired_binary",
            conditions=["baseline", "guarded"],
            model_provider="openai_responses",
            model_names=["gpt-5-mini"],
            task_count=160,
            planned_episode_count=640,
            maximum_model_calls=320,
        )