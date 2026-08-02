from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_frozen_controlled_fault_plan.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_frozen_controlled_fault_plan",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paid_launch_requires_lock(tmp_path: Path) -> None:
    module = _load_script_module()

    with pytest.raises(
        SystemExit,
        match="validated --launch-lock is required",
    ):
        module._validate_paid_launch(
            launch_lock_path=None,
            repository_root=tmp_path,
            plan_path=tmp_path / "plan.json",
            run_dir=tmp_path / "run",
            model_name="gpt-5-mini",
            max_output_tokens=2000,
            final_run=False,
        )


def test_final_launch_requires_lock(tmp_path: Path) -> None:
    module = _load_script_module()

    with pytest.raises(
        SystemExit,
        match="validated --launch-lock is required",
    ):
        module._validate_paid_launch(
            launch_lock_path=None,
            repository_root=tmp_path,
            plan_path=tmp_path / "plan.json",
            run_dir=tmp_path / "run",
            model_name="gpt-5-mini",
            max_output_tokens=2000,
            final_run=True,
        )


def test_paid_launch_rejects_nonlocked_status(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps({"status": "REFUSED"}),
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
        match="status is not LOCKED",
    ):
        module._validate_paid_launch(
            launch_lock_path=lock_path,
            repository_root=tmp_path,
            plan_path=tmp_path / "plan.json",
            run_dir=tmp_path / "run",
            model_name="gpt-5-mini",
            max_output_tokens=2000,
            final_run=False,
        )


def test_paid_launch_rejects_validation_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "audit_path": str(tmp_path / "audit.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "validate_launch_lock",
        lambda *args, **kwargs: ["Launch-lock mismatch for git."],
    )

    with pytest.raises(
        SystemExit,
        match="Launch-lock validation failed",
    ):
        module._validate_paid_launch(
            launch_lock_path=lock_path,
            repository_root=tmp_path,
            plan_path=tmp_path / "plan.json",
            run_dir=tmp_path / "run",
            model_name="gpt-5-mini",
            max_output_tokens=2000,
            final_run=True,
        )


def test_paid_launch_accepts_valid_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    lock_path = tmp_path / "lock.json"
    lock = {
        "status": "LOCKED",
        "audit_path": str(tmp_path / "audit.json"),
        "launch_lock_sha256": "abc123",
        "git": {"commit_sha": "deadbeef"},
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "validate_launch_lock",
        lambda *args, **kwargs: [],
    )

    accepted = module._validate_paid_launch(
        launch_lock_path=lock_path,
        repository_root=tmp_path,
        plan_path=tmp_path / "plan.json",
        run_dir=tmp_path / "run",
        model_name="gpt-5-mini",
        max_output_tokens=2000,
        final_run=True,
    )

    assert accepted == lock
