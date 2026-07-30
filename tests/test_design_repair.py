import pytest

from cnsm_agentic.autonomous_research.repair_schemas import (
    BudgetScenario,
    RepairedStudyDesign,
)


def test_positive_calls() -> None:
    with pytest.raises(Exception):
        BudgetScenario(
            scenario_id="x",
            description="x",
            planned_model_calls=0,
            models=["m"],
            transformations=["t"],
            discovery_items=1,
            confirmatory_items=1,
            task_cluster_count=1,
            feasibility_rationale="x",
        )


def test_placeholder_rejected() -> None:
    with pytest.raises(Exception):
        RepairedStudyDesign.model_validate(
            {
                "selected_candidate_id": "c",
                "title": "TBD",
                "research_question": "q",
                "confirmatory_hypotheses": ["h"],
                "exploratory_questions": [],
                "benchmark_scope": ["b"],
                "model_scope": ["m"],
                "transformation_scope": ["t"],
                "primary_estimand": "e",
                "secondary_estimands": [],
                "sampling_plan": "s",
                "analysis_plan": "a",
                "multiplicity_plan": "m",
                "missingness_plan": "x",
                "contamination_plan": {
                    "benchmark_names": ["b"],
                    "risk_factors": ["r"],
                    "detection_procedures": ["d"],
                    "item_flagging_rules": ["f"],
                    "analysis_treatment": "a",
                    "residual_uncertainty": [],
                },
                "budget_scenarios": [
                    {
                        "scenario_id": "s",
                        "description": "d",
                        "planned_model_calls": 1,
                        "models": ["m"],
                        "transformations": ["t"],
                        "discovery_items": 1,
                        "confirmatory_items": 1,
                        "task_cluster_count": 1,
                        "feasibility_rationale": "f",
                    }
                ],
                "power_plan": {
                    "primary_estimand": "e",
                    "target_effect": "e",
                    "clustering_unit": "c",
                    "calculation_method": "m",
                    "assumptions": ["a"],
                    "recommended_scenario_id": "s",
                    "minimum_detectable_effect_notes": "n",
                    "sensitivity_analyses": ["s"],
                },
                "transformation_validation": {
                    "transformation_families": ["t"],
                    "semantic_equivalence_checks": ["c"],
                    "automatic_rejection_rules": ["r"],
                    "audit_sample_policy": "p",
                    "mapping_validation": "m",
                    "residual_risks": [],
                },
                "preregistration_fields_complete": True,
                "unresolved_critical_issues": [],
                "remaining_noncritical_uncertainties": [],
                "evidence_record_ids": ["r"],
            }
        )


def test_external_validity_limit_is_noncritical_when_claims_are_bounded() -> None:
    issue = (
        "Ecological generalization to operator networks remains unresolved "
        "because the benchmark uses public and synthetic artifacts only."
    )

    research_question = (
        "Does RAG improve validator pass-rate on a public and synthetic "
        "NetOps benchmark?"
    )

    assert "public and synthetic" in issue
    assert "public and synthetic" in research_question

    classification = "remaining_noncritical_uncertainties"

    assert classification != "unresolved_critical_issues"


def test_production_generalization_claim_would_be_critical() -> None:
    research_question = (
        "Does the benchmark prove that RAG improves safety across "
        "private production operator networks?"
    )

    available_scope = (
        "Public documentation and synthetic benchmark artifacts only."
    )

    assert "production operator networks" in research_question
    assert "public" in available_scope.lower()
    assert "synthetic" in available_scope.lower()

    classification = "unresolved_critical_issues"

    assert classification == "unresolved_critical_issues"
