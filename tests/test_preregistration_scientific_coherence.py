from types import SimpleNamespace

from cnsm_agentic.autonomous_research.final_pipeline import (
    preregistration_identifiability_issues,
    preregistration_scientific_coherence_issues,
)


def _prereg(
    *,
    hypothesis=(
        "Guarded validation and repair reduce actionable failures."
    ),
    estimand=(
        "paired success-rate difference guarded minus baseline"
    ),
    estimand_id=(
        "paired_success_rate_difference_guarded_minus_baseline"
    ),
    analysis="Exact McNemar test with paired bootstrap.",
):
    return SimpleNamespace(
        research_question="Does guarded validation improve success?",
        confirmatory_hypotheses=[hypothesis],
        primary_estimand=estimand,
        primary_estimand_id=estimand_id,
        analysis_plan=analysis,
        transformation_scope=[],
        sampling_plan=(
            "One shared initial generation per task is reused by both arms."
        ),
        execution_contract=SimpleNamespace(
            model_dump=lambda: {
                "generation_semantics": "shared_initial_candidate",
                "initial_generation_calls_per_task": 1,
                "conditions": ["baseline", "guarded"],
                "maximum_repair_calls_per_task": 1,
            }
        ),
    )


def test_predictive_hypothesis_with_paired_difference_is_rejected():
    prereg = _prereg(
        hypothesis=(
            "A multidimensional validator metric suite predicts "
            "end-to-end success better and explains more variance "
            "than a single verifier pass rate."
        )
    )

    issues = preregistration_scientific_coherence_issues(prereg)

    assert issues


def test_difference_hypothesis_and_difference_estimand_are_coherent():
    prereg = _prereg(
        hypothesis=(
            "Guarded validation and repair increase paired success "
            "relative to baseline."
        )
    )

    issues = preregistration_scientific_coherence_issues(prereg)

    assert issues == []


def test_shared_single_draw_repair_estimand_is_structurally_identifiable():
    prereg = _prereg()

    issues = preregistration_identifiability_issues(prereg)

    assert issues == []


def test_repair_estimand_without_repair_capability_is_rejected():
    prereg = _prereg()

    prereg.execution_contract = SimpleNamespace(
        model_dump=lambda: {
            "generation_semantics": "shared_initial_candidate",
            "initial_generation_calls_per_task": 1,
            "conditions": ["baseline", "guarded"],
            "maximum_repair_calls_per_task": 0,
        }
    )

    issues = preregistration_identifiability_issues(prereg)

    assert issues


def test_repair_estimand_without_guarded_condition_is_rejected():
    prereg = _prereg()

    prereg.execution_contract = SimpleNamespace(
        model_dump=lambda: {
            "generation_semantics": "shared_initial_candidate",
            "initial_generation_calls_per_task": 1,
            "conditions": ["baseline"],
            "maximum_repair_calls_per_task": 1,
        }
    )

    issues = preregistration_identifiability_issues(prereg)

    assert issues
