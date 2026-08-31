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
