from types import SimpleNamespace

from cnsm_agentic.autonomous_research.final_pipeline import (
    canonicalize_preregistration_execution_contract,
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
    generation_semantics="shared_initial_candidate",
    independent_condition_generation=False,
    initial_generation_calls_per_task=1,
    maximum_repair_calls_per_task=1,
    retrieval_augmented_generation=False,
    research_question=(
        "Does deterministic validation and repair "
        "improve network configuration correctness?"
    ),
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
            generation_semantics=(
                generation_semantics
            ),
            independent_condition_generation=(
                independent_condition_generation
            ),
            initial_generation_calls_per_task=(
                initial_generation_calls_per_task
            ),
            maximum_repair_calls_per_task=(
                maximum_repair_calls_per_task
            ),
            retrieval_augmented_generation=(
                retrieval_augmented_generation
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
        title=(
            "Deterministic validation and repair "
            "for NetOps configuration generation"
        ),
        research_question=research_question,
        primary_estimand=(
            "Paired success-rate difference between "
            "guarded and baseline conditions."
        ),
        sampling_plan=(
            "Execute both scored conditions for each "
            "task using the registered adapter."
        ),
        analysis_plan=(
            "Use paired binary analysis."
        ),
        confirmatory_hypotheses=[
            (
                "The guarded workflow improves "
                "configuration correctness."
            ),
        ],
        benchmark_scope=[
            "Synthetic NetOps configuration tasks."
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
        "generation_semantics": (
            "shared_initial_candidate"
        ),
        "independent_condition_generation": False,
        "initial_generation_calls_per_task": 1,
        "maximum_repair_calls_per_task": 1,
        "retrieval_augmented_generation": False,
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


def test_independent_generation_claim_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            independent_condition_generation=True,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "independent_condition_generation" in issue
        for issue in issues
    )


def test_wrong_generation_semantics_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            generation_semantics=(
                "independent_condition_candidates"
            ),
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "generation_semantics" in issue
        for issue in issues
    )


def test_wrong_initial_generation_call_count_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            initial_generation_calls_per_task=2,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "initial_generation_calls_per_task"
        in issue
        for issue in issues
    )


def test_wrong_maximum_repair_call_count_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            maximum_repair_calls_per_task=2,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "maximum_repair_calls_per_task"
        in issue
        for issue in issues
    )


def test_rag_execution_contract_claim_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            retrieval_augmented_generation=True,
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "retrieval_augmented_generation"
        in issue
        for issue in issues
    )


def test_rag_prose_claim_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            research_question=(
                "Does RAG + deterministic validation "
                "improve network configuration correctness?"
            ),
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "retrieval-augmented generation"
        in issue
        for issue in issues
    )


def test_independent_generation_prose_claim_is_rejected():
    issues = preregistration_execution_contract_issues(
        _preregistration(
            research_question=(
                "Does independently generated guarded "
                "configuration output improve correctness?"
            ),
        ),
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "independent per-condition generation"
        in issue
        for issue in issues
    )

def test_canonicalization_repairs_mechanical_contract_fields():
    prereg = _preregistration(
        models=["other-model"],
        task_count=160,
        planned_episode_count=2,
        maximum_model_calls=1,
    )

    prereg.model_scope = ["other-model"]
    prereg.transformation_scope = [
        "wrong-transformation"
    ]

    canonical = (
        canonicalize_preregistration_execution_contract(
            prereg,
            planning_contracts=CONTRACTS,
            available_execution_models=[
                "gpt-5-mini"
            ],
            required_task_count=160,
        )
    )

    assert canonical.execution_contract.model_names == [
        "gpt-5-mini"
    ]
    assert canonical.model_scope == [
        "gpt-5-mini"
    ]
    assert canonical.transformation_scope == [
        "direct_configuration_generation_v1",
        "generate_validate_repair_v1",
    ]
    assert canonical.execution_contract.task_count == 160
    assert (
        canonical.execution_contract.planned_episode_count
        == 320
    )
    assert (
        canonical.execution_contract.maximum_model_calls
        == 320
    )
    assert (
        canonical.execution_contract.generation_semantics
        == "shared_initial_candidate"
    )
    assert (
        canonical.execution_contract.independent_condition_generation
        is False
    )
    assert (
        canonical.execution_contract.retrieval_augmented_generation
        is False
    )

def test_canonicalization_does_not_hide_rag_prose_mismatch():
    prereg = _preregistration(
        research_question=(
            "Does RAG + deterministic validation "
            "improve configuration correctness?"
        ),
    )

    prereg = (
        canonicalize_preregistration_execution_contract(
            prereg,
            planning_contracts=CONTRACTS,
            available_execution_models=[
                "gpt-5-mini"
            ],
            required_task_count=160,
        )
    )

    issues = preregistration_execution_contract_issues(
        prereg,
        planning_contracts=CONTRACTS,
        available_execution_models=[
            "gpt-5-mini"
        ],
        required_task_count=160,
    )

    assert any(
        "retrieval-augmented generation"
        in issue
        for issue in issues
    )