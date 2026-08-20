from cnsm_agentic.autonomous_research.execution_adapters import (
    adapter_compatibility_issues,
    clear_registered_adapters,
    register_builtin_execution_adapters,
)
from cnsm_agentic.autonomous_research.final_schemas import (
    ExperimentPlan,
)


def test_experiment_plan_can_represent_hosted_adapter_contract():
    clear_registered_adapters()
    register_builtin_execution_adapters()

    plan = ExperimentPlan(
        study_id="test-study",
        adapter_family="hosted_netops_gvr_v1",

        execution_mode="scientific_pilot",
        design="paired_binary",
        conditions=["baseline", "guarded"],
        task_families=[
            "intent_configuration_repair_v1"
        ],
        transformations={
            "baseline": (
                "direct_configuration_generation_v1"
            ),
            "guarded": (
                "generate_validate_repair_v1"
            ),
        },

        result_schema_id="paired_binary_episode_v1",
        result_schema_version="1.0",

        model_provider="openai_responses",
        model_name="gpt-5-mini",
        model_version="gpt-5-mini",

        deterministic_automated_scoring=True,
        requires_human_scientific_labour=False,

        task_count=10,
        task_indices=list(range(1, 11)),

        estimated_model_calls=20,
        maximum_model_calls=20,

        reasoning_effort="minimal",
        maximum_attempts_per_call=1,
        max_output_tokens=1000,

        implementation_strategy="Hosted paired pilot.",
        public_resources=[],
        model_and_version_plan=["gpt-5-mini"],
        task_manifest_strategy="Deterministic.",
        transformation_manifest_strategy="Deterministic.",
        execution_batches=["Single paired batch."],
        randomisation_plan="Fixed task indices.",
        caching_plan="Deterministic cache.",
        failure_recovery_plan="No provider retries.",
        result_schema="Paired binary episode.",
        estimated_compute_notes="CPU plus hosted API.",
    )

    issues = adapter_compatibility_issues(
        plan.model_dump()
    )

    assert issues == []
