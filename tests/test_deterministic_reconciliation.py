import json
from pathlib import Path

from cnsm_agentic.autonomous_research.final_pipeline import (
    build_deterministic_reconciliation,
)


def test_reconciliation_accepts_consistent_paired_counts(
    tmp_path: Path,
) -> None:
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()

    log_path = execution_dir / "execution_log.jsonl"

    events = [
        {
            "event_type": "hosted_model_call",
            "condition": "shared",
            "stage": "shared_initial_generation",
            "outcome": "COMPLETED",
            "cache_status": "MISS",
            "cache_key_sha256": "key-1",
        },
        {
            "event_type": "hosted_model_call",
            "condition": "shared",
            "stage": "shared_initial_generation",
            "outcome": "COMPLETED",
            "cache_status": "MISS",
            "cache_key_sha256": "key-2",
        },
        {
            "event_type": "hosted_model_call",
            "condition": "guarded",
            "stage": "repair",
            "outcome": "COMPLETED",
            "cache_status": "MISS",
            "cache_key_sha256": "key-3",
        },
    ]

    log_path.write_text(
        "\n".join(
            json.dumps(event)
            for event in events
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_deterministic_reconciliation(
        experiment_plan={
            "task_count": 2,
            "conditions": [
                "baseline",
                "guarded",
            ],
        },
        execution_manifest={
            "study_id": "study-1",
            "planned_episode_count": 4,
            "completed_episode_count": 4,
            "failed_episode_count": 0,
            "execution_log_path": (
                "execution/execution_log.jsonl"
            ),
        },
        analysis_results={
            "confirmatory_results": [
                {
                    "complete_pair_count": 2,
                    "baseline_success_count": 1,
                    "guarded_success_count": 2,
                    "n_11": 1,
                    "n_10": 1,
                    "n_01": 0,
                    "n_00": 0,
                }
            ],
            "missingness_summary": {
                "complete_pairs": 2,
                "baseline_only_observed_pairs": 0,
                "guarded_only_observed_pairs": 0,
                "both_missing_pairs": 0,
            },
        },
        run_dir=tmp_path,
    )

    assert (
        result[
            "marginal_contingency_consistent"
        ]
        is True
    )

    assert (
        result[
            "pair_accounting_consistent"
        ]
        is True
    )

    assert (
        result[
            "episode_accounting_consistent"
        ]
        is True
    )

    audit = result["provider_call_audit"]

    assert audit["cache_hit_count"] == 0
    assert audit["cache_miss_count"] == 3

    assert (
        audit[
            "cross_condition_cache_key_reuse_observed"
        ]
        is False
    )

    assert (
        result[
            "all_deterministic_consistency_checks_passed"
        ]
        is True
    )


def test_reconciliation_detects_marginal_mismatch(
    tmp_path: Path,
) -> None:
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()

    (
        execution_dir / "execution_log.jsonl"
    ).write_text(
        "",
        encoding="utf-8",
    )

    result = build_deterministic_reconciliation(
        experiment_plan={
            "task_count": 2,
            "conditions": [
                "baseline",
                "guarded",
            ],
        },
        execution_manifest={
            "study_id": "study-1",
            "planned_episode_count": 4,
            "completed_episode_count": 4,
            "failed_episode_count": 0,
            "execution_log_path": (
                "execution/execution_log.jsonl"
            ),
        },
        analysis_results={
            "confirmatory_results": [
                {
                    "complete_pair_count": 2,
                    "baseline_success_count": 2,
                    "guarded_success_count": 2,
                    "n_11": 1,
                    "n_10": 1,
                    "n_01": 0,
                    "n_00": 0,
                }
            ],
            "missingness_summary": {
                "complete_pairs": 2,
                "baseline_only_observed_pairs": 0,
                "guarded_only_observed_pairs": 0,
                "both_missing_pairs": 0,
            },
        },
        run_dir=tmp_path,
    )

    assert (
        result[
            "marginal_contingency_consistent"
        ]
        is False
    )

    assert (
        result[
            "all_deterministic_consistency_checks_passed"
        ]
        is False
    )
