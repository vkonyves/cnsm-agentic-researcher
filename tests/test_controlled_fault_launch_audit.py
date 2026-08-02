from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    generate_experiment_plan,
    write_experiment_plan,
)
from cnsm_agentic.autonomous_research.controlled_fault_launch_audit import (
    audit_controlled_fault_launch,
)


def _write_plan(tmp_path: Path, *, pair_count: int = 40) -> Path:
    plan = generate_experiment_plan(pair_count=pair_count, seed=17)
    path = tmp_path / "plan.json"
    write_experiment_plan(plan, path)
    return path


def test_launch_audit_passes_without_provider_calls(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    output_path = tmp_path / "studies" / "audits" / "audit.json"

    report = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    assert report["status"] == "PASS"
    assert report["provider_initialized"] is False
    assert report["provider_calls_made"] == 0
    assert report["pair_count"] == 40
    assert report["maximum_model_calls"] == 80
    assert report["computed_maximum_model_calls"] == 80
    assert report["source_prompt_count"] == 40
    assert report["repair_prompt_count"] == 40
    assert len(report["pairs"]) == 40
    assert all(pair["status"] == "PASS" for pair in report["pairs"])


def test_launch_audit_is_deterministic(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    output_path = tmp_path / "studies" / "audits" / "audit.json"

    first = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )
    second = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    assert first == second
    assert (
        first["aggregate_prompt_sha256"]
        == second["aggregate_prompt_sha256"]
    )
    assert (
        first["aggregate_shared_candidate_sha256"]
        == second["aggregate_shared_candidate_sha256"]
    )
    assert first["audit_report_sha256"] == second["audit_report_sha256"]


def test_launch_audit_checks_string_metadata(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    output_path = tmp_path / "studies" / "audits" / "audit.json"

    report = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    for pair in report["pairs"]:
        for metadata in pair["metadata_preview"].values():
            assert all(
                isinstance(value, str)
                for value in metadata.values()
            )
            assert metadata["task_index"].isdigit()


def test_launch_audit_rejects_unsafe_output_path(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    output_path = tmp_path / "src" / "audit.json"

    report = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    assert report["status"] == "FAIL"
    assert any(
        "must not be written under" in issue
        for issue in report["issues"]
    )


def test_launch_audit_detects_bad_model_settings(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    output_path = tmp_path / "studies" / "audits" / "audit.json"

    report = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=tmp_path,
        output_path=output_path,
        model_name=" ",
        max_output_tokens=0,
    )

    assert report["status"] == "FAIL"
    assert "max_output_tokens must be positive." in report["issues"]
    assert "model_name must be non-empty." in report["issues"]
