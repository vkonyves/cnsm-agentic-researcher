import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cnsm_agentic.autonomous_research.analysis_executors import (
    analysis_family_matches,
    clear_registered_analysis_executors,
    normalise_analysis_family,
    register_analysis_executor,
    registered_analysis_families,
    resolve_analysis_executor,
    validate_analysis_results,
    paired_binary_analysis_compatibility_issues,
    PairedBinaryAnalysisExecutor,
    register_builtin_analysis_executors,
)


class ExampleAnalysisExecutor:
    family = "paired_binary_analysis_v1"
    aliases = (
        "paired-binary-analysis-v1",
    )

    def supports(
        self,
        *,
        analysis_plan: dict[str, Any],
        execution_manifest: dict[str, Any],
    ) -> bool:
        return analysis_family_matches(
            analysis_plan.get(
                "analysis_executor"
            ),
            family=self.family,
            aliases=self.aliases,
        )

    def execute(
        self,
        *,
        analysis_plan: dict[str, Any],
        preregistration: dict[str, Any],
        execution_manifest: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_analysis_registry():
    clear_registered_analysis_executors()
    yield
    clear_registered_analysis_executors()


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_empty_analysis_registry_resolves_none() -> None:
    resolved = resolve_analysis_executor(
        analysis_plan={
            "analysis_executor": (
                "paired_binary_analysis_v1"
            )
        },
        execution_manifest={},
    )

    assert resolved is None


def test_registered_analysis_executor_resolves() -> None:
    executor = ExampleAnalysisExecutor()

    register_analysis_executor(
        executor
    )

    resolved = resolve_analysis_executor(
        analysis_plan={
            "analysis_executor": (
                "paired_binary_analysis_v1"
            )
        },
        execution_manifest={},
    )

    assert resolved is executor


def test_explicit_analysis_alias_resolves() -> None:
    executor = ExampleAnalysisExecutor()

    register_analysis_executor(
        executor
    )

    resolved = resolve_analysis_executor(
        analysis_plan={
            "analysis_executor": (
                "paired-binary-analysis-v1"
            )
        },
        execution_manifest={},
    )

    assert resolved is executor


def test_descriptive_analysis_name_does_not_match() -> None:
    register_analysis_executor(
        ExampleAnalysisExecutor()
    )

    resolved = resolve_analysis_executor(
        analysis_plan={
            "analysis_executor": (
                "Paired binary bootstrap "
                "analysis with confidence intervals"
            )
        },
        execution_manifest={},
    )

    assert resolved is None


def test_duplicate_analysis_executor_rejected() -> None:
    register_analysis_executor(
        ExampleAnalysisExecutor()
    )

    with pytest.raises(
        ValueError
    ):
        register_analysis_executor(
            ExampleAnalysisExecutor()
        )


def test_registered_analysis_families() -> None:
    register_analysis_executor(
        ExampleAnalysisExecutor()
    )

    assert registered_analysis_families() == [
        "paired_binary_analysis_v1"
    ]


def test_analysis_family_normalisation() -> None:
    assert normalise_analysis_family(
        "Paired-Binary Analysis v1"
    ) == "paired_binary_analysis_v1"


def test_valid_analysis_results_pass(
    tmp_path: Path,
) -> None:
    execution_dir = (
        tmp_path / "execution"
    )
    analysis_dir = (
        tmp_path / "analysis"
    )

    execution_dir.mkdir()
    analysis_dir.mkdir()

    raw_results = (
        execution_dir
        / "raw_results.jsonl"
    )

    raw_results.write_text(
        '{"episode_id": 1}\n',
        encoding="utf-8",
    )

    raw_hash = _sha256(
        raw_results
    )

    results_path = (
        analysis_dir
        / "results.json"
    )

    results_path.write_text(
        '{"status": "COMPLETED"}',
        encoding="utf-8",
    )

    results_hash = _sha256(
        results_path
    )

    results = {
        "status": "COMPLETED",
        "schema_version": "1.0",
        "analysis_executor": (
            "paired_binary_analysis_v1"
        ),
        "study_id": "study-1",
        "input_results_sha256": raw_hash,
        "confirmatory_results": [
            {
                "hypothesis_id": "H1",
                "estimate": 0.2,
                "sample_size": 100,
            }
        ],
        "secondary_results": [],
        "missingness_summary": {},
        "exclusions": [],
        "deviations_from_preregistration": [],
        "results_path": (
            "analysis/results.json"
        ),
        "artifact_hashes": {
            "analysis/results.json": (
                results_hash
            )
        },
        "warnings": [],
    }

    execution_manifest = {
        "results_path": (
            "execution/raw_results.jsonl"
        ),
        "artifact_hashes": {
            "execution/raw_results.jsonl": (
                raw_hash
            )
        },
    }

    issues = validate_analysis_results(
        results,
        run_dir=tmp_path,
        execution_manifest=(
            execution_manifest
        ),
    )

    assert issues == []


def test_analysis_input_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    analysis_dir = (
        tmp_path / "analysis"
    )
    analysis_dir.mkdir()

    results_path = (
        analysis_dir
        / "results.json"
    )

    results_path.write_text(
        "{}",
        encoding="utf-8",
    )

    results = {
        "status": "COMPLETED",
        "schema_version": "1.0",
        "analysis_executor": (
            "paired_binary_analysis_v1"
        ),
        "study_id": "study-1",
        "input_results_sha256": "wrong",
        "confirmatory_results": [
            {
                "hypothesis_id": "H1"
            }
        ],
        "secondary_results": [],
        "missingness_summary": {},
        "exclusions": [],
        "deviations_from_preregistration": [],
        "results_path": (
            "analysis/results.json"
        ),
        "artifact_hashes": {
            "analysis/results.json": (
                _sha256(
                    results_path
                )
            )
        },
        "warnings": [],
    }

    issues = validate_analysis_results(
        results,
        run_dir=tmp_path,
        execution_manifest={
            "results_path": (
                "execution/raw_results.jsonl"
            ),
            "artifact_hashes": {
                "execution/raw_results.jsonl": (
                    "expected"
                )
            },
        },
    )

    assert any(
        "input hash"
        in issue.lower()
        for issue in issues
    )


def test_completed_analysis_requires_results(
    tmp_path: Path,
) -> None:
    issues = validate_analysis_results(
        {
            "status": "COMPLETED",
            "confirmatory_results": [],
        },
        run_dir=tmp_path,
        execution_manifest={},
    )

    assert any(
        "no confirmatory results"
        in issue.lower()
        for issue in issues
    )


def test_analysis_path_traversal_rejected(
    tmp_path: Path,
) -> None:
    issues = validate_analysis_results(
        {
            "status": "FAILED",
            "results_path": "../outside.json",
        },
        run_dir=tmp_path,
        execution_manifest={},
    )

    assert any(
        "unsafe"
        in issue.lower()
        for issue in issues
    )



def test_paired_analysis_compatibility_passes() -> None:
    issues = paired_binary_analysis_compatibility_issues(
        analysis_plan={
            "analysis_executor": "paired_binary_analysis_v1",
            "study_id": "study-1",
            "estimand": (
                "paired_success_rate_difference_guarded_minus_baseline"
            ),
            "failed_call_treatment": "complete_pair_primary",
        },
        execution_manifest={
            "status": "COMPLETED",
            "adapter_family": "synthetic_paired_llm_benchmark_v1",
            "study_id": "study-1",
            "result_schema_id": "paired_binary_episode_v1",
            "result_schema_version": "1.0",
            "results_path": "execution/raw_results.jsonl",
            "result_schema_path": "execution/result_schema.json",
            "artifact_hashes": {"execution/raw_results.jsonl": "x"},
            "execution_mode": "development_rehearsal",
        },
    )
    assert issues == []


def test_paired_analysis_rejects_mismatched_study() -> None:
    issues = paired_binary_analysis_compatibility_issues(
        analysis_plan={
            "analysis_executor": "paired_binary_analysis_v1",
            "study_id": "study-2",
            "estimand": (
                "paired_success_rate_difference_guarded_minus_baseline"
            ),
            "failed_call_treatment": "complete_pair_primary",
        },
        execution_manifest={
            "status": "COMPLETED",
            "adapter_family": "synthetic_paired_llm_benchmark_v1",
            "study_id": "study-1",
            "result_schema_id": "paired_binary_episode_v1",
            "result_schema_version": "1.0",
            "results_path": "execution/raw_results.jsonl",
            "result_schema_path": "execution/result_schema.json",
            "artifact_hashes": {"execution/raw_results.jsonl": "x"},
        },
    )
    assert any("study ids" in issue.lower() for issue in issues)


def test_final_analysis_requires_master_prompt_provenance() -> None:
    issues = paired_binary_analysis_compatibility_issues(
        analysis_plan={
            "analysis_executor": "paired_binary_analysis_v1",
            "study_id": "study-1",
            "estimand": (
                "paired_success_rate_difference_guarded_minus_baseline"
            ),
            "failed_call_treatment": "complete_pair_primary",
        },
        execution_manifest={
            "status": "COMPLETED",
            "adapter_family": "synthetic_paired_llm_benchmark_v1",
            "study_id": "study-1",
            "result_schema_id": "paired_binary_episode_v1",
            "result_schema_version": "1.0",
            "results_path": "execution/raw_results.jsonl",
            "result_schema_path": "execution/result_schema.json",
            "artifact_hashes": {"execution/raw_results.jsonl": "x"},
            "execution_mode": "final_autonomous_run",
            "human_scientific_intervention_after_launch": False,
        },
    )
    assert any("master-prompt" in issue.lower() for issue in issues)



def test_builtin_analysis_registration_is_explicit_and_idempotent() -> None:
    register_builtin_analysis_executors()
    register_builtin_analysis_executors()
    assert registered_analysis_families() == ["paired_binary_analysis_v1"]


def test_end_to_end_paired_analysis_rehearsal(tmp_path: Path) -> None:
    from cnsm_agentic.autonomous_research.execution_adapters import (
        SyntheticPairedLLMBenchmarkAdapter,
        validate_execution_manifest,
    )

    task_count = 6
    plan = {
        "study_id": "study-analysis-1",
        "adapter_family": "synthetic_paired_llm_benchmark_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "task_count": task_count,
        "estimated_model_calls": task_count * 2,
        "task_families": ["configuration_error_detection_v1"],
        "transformations": {
            "baseline": "baseline_prompt_v1",
            "guarded": "guarded_prompt_v1",
        },
        "model_provider": "deterministic_local",
        "model_name": "paired-smoke-model",
        "model_version": "1.0",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "development_rehearsal",
    }
    manifest = SyntheticPairedLLMBenchmarkAdapter().execute(
        plan=plan,
        preregistration={"study_id": "study-analysis-1"},
        output_dir=tmp_path / "execution",
    )
    assert validate_execution_manifest(
        manifest,
        plan=plan,
        output_dir=tmp_path / "execution",
        maximum_model_calls=12,
    ) == []

    analysis_plan = {
        "analysis_executor": "paired_binary_analysis_v1",
        "study_id": "study-analysis-1",
        "estimand": "paired_success_rate_difference_guarded_minus_baseline",
        "failed_call_treatment": "complete_pair_primary",
        "bootstrap_seed": 7,
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
    }
    executor = PairedBinaryAnalysisExecutor()
    assert executor.supports(
        analysis_plan=analysis_plan,
        execution_manifest=manifest,
    )
    results = executor.execute(
        analysis_plan=analysis_plan,
        preregistration={"study_id": "study-analysis-1"},
        execution_manifest=manifest,
        run_dir=tmp_path,
    )
    primary = results["confirmatory_results"][0]
    assert primary["complete_pair_count"] == 6
    assert primary["n_10"] == 2
    assert primary["n_01"] == 0
    assert primary["estimate"] == pytest.approx(2 / 6)
    assert results["input_results_sha256"] == manifest["artifact_hashes"][
        manifest["results_path"]
    ]
    assert validate_analysis_results(
        results,
        run_dir=tmp_path,
        execution_manifest=manifest,
    ) == []
    persisted_results = json.loads(
        (tmp_path / results["results_path"]).read_text(encoding="utf-8")
    )
    assert persisted_results == results
    assert validate_analysis_results(
        persisted_results,
        run_dir=tmp_path,
        execution_manifest=manifest,
    ) == []
    assert results["results_path"] not in results["artifact_hashes"]
    for relative_path in results["artifact_hashes"]:
        assert (tmp_path / relative_path).is_file()


def test_analysis_reports_incomplete_pair_from_failed_call(tmp_path: Path) -> None:
    from cnsm_agentic.autonomous_research.execution_adapters import (
        SyntheticPairedLLMBenchmarkAdapter,
    )

    task_count = 3
    plan = {
        "study_id": "study-missing-1",
        "adapter_family": "synthetic_paired_llm_benchmark_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "task_count": task_count,
        "estimated_model_calls": task_count * 2,
        "task_families": ["configuration_error_detection_v1"],
        "transformations": {
            "baseline": "baseline_prompt_v1",
            "guarded": "guarded_prompt_v1",
        },
        "model_provider": "deterministic_local",
        "model_name": "paired-smoke-model",
        "model_version": "1.0",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "development_rehearsal",
        "rehearsal_failure_task_ids": ["task-000001"],
    }
    manifest = SyntheticPairedLLMBenchmarkAdapter().execute(
        plan=plan,
        preregistration={},
        output_dir=tmp_path / "execution",
    )
    results = PairedBinaryAnalysisExecutor().execute(
        analysis_plan={
            "analysis_executor": "paired_binary_analysis_v1",
            "study_id": "study-missing-1",
            "estimand": "paired_success_rate_difference_guarded_minus_baseline",
            "failed_call_treatment": "complete_pair_primary",
            "bootstrap_resamples": 100,
        },
        preregistration={},
        execution_manifest=manifest,
        run_dir=tmp_path,
    )
    assert results["missingness_summary"]["guarded_only_observed_pairs"] == 1
    assert results["missingness_summary"]["failed_baseline_episodes"] == 1
    assert results["confirmatory_results"][0]["complete_pair_count"] == 2
