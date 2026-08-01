import hashlib
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
