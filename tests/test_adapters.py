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