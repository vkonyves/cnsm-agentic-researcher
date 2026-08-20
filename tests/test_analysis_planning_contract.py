from cnsm_agentic.autonomous_research.analysis_executors import (
    analysis_compatibility_issues,
    clear_registered_analysis_executors,
    register_builtin_analysis_executors,
    registered_analysis_planning_contracts,
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
