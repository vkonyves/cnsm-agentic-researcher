import pytest

from cnsm_agentic.autonomous_research.final_schemas import (
    AnalysisPlan,
    ExperimentPlan,
    PreregistrationDocument,
)


def test_zero_calls_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentPlan(
            study_id="s",
            adapter_family="x",
            implementation_strategy="i",
            public_resources=["r"],
            model_and_version_plan=["m"],
            task_manifest_strategy="t",
            transformation_manifest_strategy="x",
            execution_batches=["b"],
            randomisation_plan="r",
            caching_plan="c",
            failure_recovery_plan="f",
            result_schema="s",
            estimated_model_calls=0,
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
            model_scope=["m"],
            transformation_scope=["t"],
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
            "paired_success_rate_difference_guarded_minus_baseline"
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
            "No multiplicity adjustment for one primary estimand."
        ),
        contamination_analysis=(
            "Report deterministic contamination flags."
        ),
        failed_call_treatment=(
            "Report failures and apply the preregistered rule."
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
            primary_analysis="Primary analysis.",
            secondary_analyses=[],
            sensitivity_analyses=[],
            uncertainty_quantification="Confidence interval.",
            multiplicity_implementation="None.",
            contamination_analysis="None.",
            failed_call_treatment="Report failures.",
            table_specifications=[],
            figure_specifications=[],
        )