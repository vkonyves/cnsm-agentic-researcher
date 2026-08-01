import hashlib
from pathlib import Path
from typing import Any

import pytest

from cnsm_agentic.autonomous_research.execution_adapters import (
    adapter_family_matches,
    clear_registered_adapters,
    normalise_adapter_family,
    register_adapter,
    registered_adapter_families,
    resolve_adapter,
    validate_execution_manifest,
    validate_final_autonomy_contract,
    validate_paired_binary_result_row,
    synthetic_paired_plan_issues,
)


class ExampleAdapter:
    family = "synthetic_paired_llm_benchmark_v1"
    aliases = (
        "synthetic-paired-llm-benchmark-v1",
    )

    def supports(
        self,
        plan: dict[str, Any],
    ) -> bool:
        return adapter_family_matches(
            plan,
            family=self.family,
            aliases=self.aliases,
        )

    def execute(
        self,
        *,
        plan: dict[str, Any],
        preregistration: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_adapter_registry():
    clear_registered_adapters()
    yield
    clear_registered_adapters()


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_empty_registry_resolves_none() -> None:
    assert resolve_adapter(
        {
            "adapter_family": (
                "synthetic_paired_llm_benchmark_v1"
            )
        }
    ) is None


def test_registered_adapter_resolves_exact_family() -> None:
    adapter = ExampleAdapter()

    register_adapter(
        adapter
    )

    resolved = resolve_adapter(
        {
            "adapter_family": (
                "synthetic_paired_llm_benchmark_v1"
            )
        }
    )

    assert resolved is adapter


def test_registered_adapter_resolves_explicit_alias() -> None:
    adapter = ExampleAdapter()

    register_adapter(
        adapter
    )

    resolved = resolve_adapter(
        {
            "adapter_family": (
                "synthetic-paired-llm-benchmark-v1"
            )
        }
    )

    assert resolved is adapter


def test_descriptive_family_does_not_fuzzy_match() -> None:
    register_adapter(
        ExampleAdapter()
    )

    resolved = resolve_adapter(
        {
            "adapter_family": (
                "Synthetic paired LLM benchmark "
                "with hosted API and CPU scoring"
            )
        }
    )

    assert resolved is None


def test_duplicate_adapter_identifier_rejected() -> None:
    register_adapter(
        ExampleAdapter()
    )

    with pytest.raises(
        ValueError
    ):
        register_adapter(
            ExampleAdapter()
        )


def test_registered_families_are_reported() -> None:
    register_adapter(
        ExampleAdapter()
    )

    assert registered_adapter_families() == [
        "synthetic_paired_llm_benchmark_v1"
    ]


def test_adapter_family_normalisation() -> None:
    assert normalise_adapter_family(
        "Synthetic-Paired LLM Benchmark v1"
    ) == (
        "synthetic_paired_llm_benchmark_v1"
    )


def test_valid_completed_manifest_passes(
    tmp_path: Path,
) -> None:
    output_dir = (
        tmp_path / "execution"
    )
    output_dir.mkdir()

    results_path = (
        output_dir / "raw_results.jsonl"
    )
    schema_path = (
        output_dir / "result_schema.json"
    )
    log_path = (
        output_dir / "execution_log.jsonl"
    )

    results_path.write_text(
        '{"episode_id": 1}\n',
        encoding="utf-8",
    )
    schema_path.write_text(
        '{"type": "object"}',
        encoding="utf-8",
    )
    log_path.write_text(
        '{"event": "completed"}\n',
        encoding="utf-8",
    )

    manifest = {
        "status": "COMPLETED",
        "schema_version": "1.0",
        "adapter_family": (
            "synthetic_paired_llm_benchmark_v1"
        ),
        "study_id": "study-1",
        "started_at_utc": (
            "2026-08-01T12:00:00+00:00"
        ),
        "completed_at_utc": (
            "2026-08-01T12:10:00+00:00"
        ),
        "planned_episode_count": 1,
        "completed_episode_count": 1,
        "failed_episode_count": 0,
        "model_calls_used": 2,
        "results_path": "raw_results.jsonl",
        "result_schema_path": (
            "result_schema.json"
        ),
        "execution_log_path": (
            "execution_log.jsonl"
        ),
        "artifact_hashes": {
            "raw_results.jsonl": (
                _sha256(
                    results_path
                )
            ),
            "result_schema.json": (
                _sha256(
                    schema_path
                )
            ),
            "execution_log.jsonl": (
                _sha256(
                    log_path
                )
            ),
        },
        "warnings": [],
    }

    issues = validate_execution_manifest(
        manifest,
        plan={
            "adapter_family": (
                "synthetic_paired_llm_benchmark_v1"
            )
        },
        output_dir=output_dir,
        maximum_model_calls=10,
    )

    assert issues == []


def test_completed_status_without_artifacts_fails(
    tmp_path: Path,
) -> None:
    output_dir = (
        tmp_path / "execution"
    )
    output_dir.mkdir()

    issues = validate_execution_manifest(
        {
            "status": "COMPLETED",
        },
        plan={
            "adapter_family": "x"
        },
        output_dir=output_dir,
        maximum_model_calls=10,
    )

    assert any(
        "lacks required field"
        in issue.lower()
        for issue in issues
    )


def test_manifest_rejects_count_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = (
        tmp_path / "execution"
    )
    output_dir.mkdir()

    issues = validate_execution_manifest(
        {
            "status": "FAILED",
            "planned_episode_count": 10,
            "completed_episode_count": 7,
            "failed_episode_count": 1,
            "model_calls_used": 0,
        },
        plan={
            "adapter_family": "x"
        },
        output_dir=output_dir,
    )

    assert any(
        "do not equal"
        in issue.lower()
        for issue in issues
    )


def test_manifest_rejects_model_call_overrun(
    tmp_path: Path,
) -> None:
    output_dir = (
        tmp_path / "execution"
    )
    output_dir.mkdir()

    issues = validate_execution_manifest(
        {
            "status": "FAILED",
            "planned_episode_count": 1,
            "completed_episode_count": 0,
            "failed_episode_count": 1,
            "model_calls_used": 11,
        },
        plan={
            "adapter_family": "x"
        },
        output_dir=output_dir,
        maximum_model_calls=10,
    )

    assert any(
        "model calls exceed"
        in issue.lower()
        for issue in issues
    )


def test_manifest_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    output_dir = (
        tmp_path / "execution"
    )
    output_dir.mkdir()

    issues = validate_execution_manifest(
        {
            "status": "FAILED",
            "planned_episode_count": 1,
            "completed_episode_count": 0,
            "failed_episode_count": 1,
            "model_calls_used": 0,
            "results_path": "../outside.json",
            "result_schema_path": (
                "../schema.json"
            ),
            "execution_log_path": (
                "../log.jsonl"
            ),
        },
        plan={
            "adapter_family": "x"
        },
        output_dir=output_dir,
    )

    assert any(
        "unsafe"
        in issue.lower()
        for issue in issues
    )


def _valid_paired_row() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "schema_version": "1.0",
        "result_schema_id": "paired_binary_episode_v1",
        "study_id": "study-1",
        "episode_id": "episode-1-baseline",
        "pair_id": "pair-1",
        "task_id": "task-1",
        "task_family": "configuration_error_detection_v1",
        "condition": "baseline",
        "paired_condition": "guarded",
        "condition_order": 1,
        "execution_mode": "development_rehearsal",
        "model_provider": "deterministic_local",
        "model_name": "paired-smoke-model",
        "model_version": "1.0",
        "model_configuration_sha256": digest,
        "task_manifest_id": "task-manifest-v1",
        "task_manifest_sha256": digest,
        "task_input_sha256": digest,
        "reference_answer_sha256": digest,
        "transformation_id": "baseline_prompt_v1",
        "transformation_manifest_sha256": digest,
        "prompt_sha256": digest,
        "call_status": "COMPLETED",
        "attempt_count": 1,
        "model_calls_used": 1,
        "terminal_error_type": None,
        "terminal_error_message": None,
        "response_sha256": digest,
        "response_artifact_path": "execution/responses/x.txt",
        "scoring_status": "COMPLETED",
        "score": 1,
        "score_reason_code": "EXACT_MATCH",
        "scorer_id": "deterministic_netops_scorer_v1",
        "scorer_version": "1.0",
        "scoring_input_sha256": digest,
        "scoring_artifact_sha256": digest,
        "contamination_flags": [],
        "validity_flags": [],
        "started_at_utc": "2026-08-01T20:00:00+00:00",
        "completed_at_utc": "2026-08-01T20:00:01+00:00",
        "latency_ms": 1000,
    }


def test_valid_paired_binary_row_passes() -> None:
    assert validate_paired_binary_result_row(_valid_paired_row()) == []


def test_failed_call_must_remain_unscored() -> None:
    row = _valid_paired_row()
    row.update({
        "call_status": "FAILED",
        "scoring_status": "COMPLETED",
        "score": 0,
        "terminal_error_type": "TimeoutError",
        "response_sha256": None,
        "response_artifact_path": None,
    })
    issues = validate_paired_binary_result_row(row)
    assert any("unscored" in issue.lower() for issue in issues)


def test_final_run_requires_sealed_autonomy_contract() -> None:
    issues = validate_final_autonomy_contract({
        "execution_mode": "final_autonomous_run",
    })
    assert any("master_prompt_sha256" in issue for issue in issues)
    assert any("prohibit human scientific" in issue.lower() for issue in issues)


def test_final_run_autonomy_contract_passes() -> None:
    digest = "b" * 64
    assert validate_final_autonomy_contract({
        "execution_mode": "final_autonomous_run",
        "master_prompt_sha256": digest,
        "framework_commit": "afc39f7",
        "framework_tag": "v1.0.0-final",
        "capability_manifest_sha256": digest,
        "preregistration_sha256": digest,
        "human_scientific_intervention_after_launch": False,
        "human_text_editing_after_launch": False,
    }) == []


def test_synthetic_plan_rejects_human_scientific_labour() -> None:
    plan = {
        "adapter_family": "synthetic_paired_llm_benchmark_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "estimated_model_calls": 2,
        "task_families": ["configuration_error_detection_v1"],
        "transformations": {
            "baseline": "baseline_prompt_v1",
            "guarded": "guarded_prompt_v1",
        },
        "model_provider": "deterministic_local",
        "model_name": "paired-smoke-model",
        "model_version": "1.0",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": True,
        "execution_mode": "development_rehearsal",
    }
    issues = synthetic_paired_plan_issues(
        plan,
        maximum_model_calls=10,
        supported_task_families=("configuration_error_detection_v1",),
        supported_transformations=(
            "baseline_prompt_v1", "guarded_prompt_v1",
        ),
        available_models=((
            "deterministic_local", "paired-smoke-model", "1.0",
        ),),
    )
    assert any("human scientific labour" in issue.lower() for issue in issues)
