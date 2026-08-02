import json
import subprocess
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    generate_experiment_plan,
    write_experiment_plan,
)
from cnsm_agentic.autonomous_research.controlled_fault_launch_audit import (
    audit_controlled_fault_launch,
    write_launch_audit,
)
from cnsm_agentic.autonomous_research.controlled_fault_launch_lock import (
    create_launch_lock,
    validate_launch_lock,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    (root / "README.md").write_text("test\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "Initial commit")
    return root


def _artifacts(root: Path) -> tuple[Path, Path]:
    plan_path = root / "studies" / "plans" / "plan.json"
    audit_path = root / "studies" / "audits" / "audit.json"
    plan_path.parent.mkdir(parents=True)
    audit_path.parent.mkdir(parents=True)

    plan = generate_experiment_plan(pair_count=40, seed=17)
    write_experiment_plan(plan, plan_path)
    audit = audit_controlled_fault_launch(
        plan_path=plan_path,
        repository_root=root,
        output_path=audit_path,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )
    write_launch_audit(audit, audit_path)
    _git(root, "add", "studies")
    _git(root, "commit", "-m", "Add frozen artifacts")
    return plan_path, audit_path


def test_launch_lock_passes_for_clean_repository(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    plan_path, audit_path = _artifacts(root)
    run_dir = root / "studies" / "experiments" / "run"

    lock = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=root,
        intended_run_dir=run_dir,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    assert lock["status"] == "LOCKED"
    assert lock["git"]["working_tree_clean"] is True
    assert lock["maximum_model_calls"] == 80
    assert lock["issues"] == []


def test_launch_lock_refuses_dirty_repository(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    plan_path, audit_path = _artifacts(root)
    (root / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    lock = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=root,
        intended_run_dir=root / "run",
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )

    assert lock["status"] == "REFUSED"
    assert "Git working tree is not clean." in lock["issues"]


def test_launch_lock_requires_requested_tag(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    plan_path, audit_path = _artifacts(root)

    refused = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=root,
        intended_run_dir=root / "run",
        model_name="gpt-5-mini",
        max_output_tokens=2000,
        require_head_tag="v-test",
    )
    assert refused["status"] == "REFUSED"

    _git(root, "tag", "v-test")
    accepted = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=root,
        intended_run_dir=root / "run",
        model_name="gpt-5-mini",
        max_output_tokens=2000,
        require_head_tag="v-test",
    )
    assert accepted["status"] == "LOCKED"


def test_launch_lock_validation_detects_repository_change(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    plan_path, audit_path = _artifacts(root)
    run_dir = root / "run"

    lock = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=root,
        intended_run_dir=run_dir,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )
    assert validate_launch_lock(
        lock,
        repository_root=root,
        plan_path=plan_path,
        audit_path=audit_path,
        intended_run_dir=run_dir,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    ) == []

    (root / "dirty.txt").write_text("changed\n", encoding="utf-8")
    issues = validate_launch_lock(
        lock,
        repository_root=root,
        plan_path=plan_path,
        audit_path=audit_path,
        intended_run_dir=run_dir,
        model_name="gpt-5-mini",
        max_output_tokens=2000,
    )
    assert issues
