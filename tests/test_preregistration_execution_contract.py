from types import SimpleNamespace

from cnsm_agentic.autonomous_research.final_pipeline import (
    preregistration_execution_contract_issues,
)
from cnsm_agentic.autonomous_research.final_schemas import (
    PreregistrationExecutionContract,
)


def _preregistration(
    *,
    conditions=None,
    models=None,
    task_count=160,
    planned_episode_count=320,
    maximum_model_calls=320,
):
    models = models or ["gpt-5-mini"]

    execution_contract = (
        PreregistrationExecutionContract(
            adapter_family="hosted_netops_gvr_v1",
            execution_mode="scientific_confirmatory",
            design="paired_binary",
            conditions=conditions
            or ["baseline", "guarded"],
            model_provider="openai_responses",
            model_names=models,
            task_count=task_count,
            planned_episode_count=(
                planned_episode_count
            ),
            maximum_model_calls=(
                maximum_model_calls
            ),
        )
    )

    return SimpleNamespace(
        execution_contract=execution_contract,
        model_scope=models,
        transformation_scope=[
            "direct_configuration_generation_v1",
            "generate_validate_repair_v1",
        ],
    )


CONTRACTS = {
    "hosted_netops_gvr_v1": {
        "adapter_family": "hosted_netops_gvr_v1",
        "execution_mode": "scientific_confirmatory",
        "design": "paired_binary",
        "conditions": ["baseline", "guarded"],
        "model_provider": "openai_responses",
        "transformations": {
            "baseline": (
                "direct_configuration_generation_v1"
            ),
            "guarded": (
                "generate_validate_repair_v1"
            ),
        },
        "episodes_per_task": 2,
        "maximum_model_calls_per_task": 2,
    }
}


def test_matching_preregistration_contract_passes():
    issues = preregistration_execution_contract_issues(
        _preregistration(),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert issues == []


def test_extra_model_family_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            models=[
                "gpt-5-mini",
                "other-model",
            ],
            planned_episode_count=320,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "model_names" in issue
        for issue in issues
    )


def test_wrong_task_count_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            task_count=80,
            planned_episode_count=160,
            maximum_model_calls=160,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "task_count" in issue
        for issue in issues
    )


def test_wrong_planned_episode_count_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            task_count=160,
            planned_episode_count=2,
            maximum_model_calls=320,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "planned_episode_count" in issue
        for issue in issues
    )