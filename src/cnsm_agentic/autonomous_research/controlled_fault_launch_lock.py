from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controlled_fault_experiment_plan import load_experiment_plan
from .hosted_controlled_fault_plan_runner import (
    PLAN_RUNNER_ID,
    PLAN_RUNNER_VERSION,
    PROMPT_PROTOCOL_VERSION,
    SUPPORTED_PROVIDER,
    _code_fingerprint,
)


LAUNCH_LOCK_ID = "controlled_fault_launch_lock_v1"
LAUNCH_LOCK_VERSION = "1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_git(
    repository_root: Path,
    *args: str,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _working_tree_status(repository_root: Path) -> str:
    return _run_git(repository_root, "status", "--porcelain=v1")


def _head_commit(repository_root: Path) -> str:
    return _run_git(repository_root, "rev-parse", "HEAD")


def _head_branch(repository_root: Path) -> str:
    return _run_git(
        repository_root,
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )


def _head_tags(repository_root: Path) -> list[str]:
    output = _run_git(
        repository_root,
        "tag",
        "--points-at",
        "HEAD",
    )
    return sorted(line for line in output.splitlines() if line)


def create_launch_lock(
    *,
    plan_path: Path,
    audit_path: Path,
    repository_root: Path,
    intended_run_dir: Path,
    model_name: str,
    max_output_tokens: int,
    require_clean_worktree: bool = True,
    require_head_tag: str | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    plan_path = plan_path.resolve()
    audit_path = audit_path.resolve()
    intended_run_dir = intended_run_dir.resolve()

    plan = load_experiment_plan(plan_path)
    audit = _read_json(audit_path)
    issues: list[str] = []

    if audit.get("status") != "PASS":
        issues.append("Launch audit status is not PASS.")
    if audit.get("provider_calls_made") != 0:
        issues.append("Launch audit unexpectedly made provider calls.")
    if audit.get("provider_initialized") is not False:
        issues.append("Launch audit unexpectedly initialized a provider.")
    if audit.get("plan_sha256") != plan["plan_sha256"]:
        issues.append("Audit plan hash does not match the frozen plan.")
    if audit.get("pair_count") != plan["pair_count"]:
        issues.append("Audit pair count does not match the frozen plan.")
    if audit.get("maximum_model_calls") != plan["maximum_model_calls"]:
        issues.append("Audit call ceiling does not match the frozen plan.")

    audit_runner = audit.get("runner", {})
    expected_runner = {
        "runner_id": PLAN_RUNNER_ID,
        "runner_version": PLAN_RUNNER_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "code_fingerprint_sha256": _code_fingerprint(),
    }
    for key, expected in expected_runner.items():
        if audit_runner.get(key) != expected:
            issues.append(
                f"Audit runner mismatch for {key}: "
                f"{audit_runner.get(key)!r} != {expected!r}."
            )

    expected_model = {
        "provider": SUPPORTED_PROVIDER,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "maximum_attempts_per_call": 1,
        "reasoning_effort": "minimal",
    }
    audit_model = audit.get("model_settings", {})
    for key, expected in expected_model.items():
        if audit_model.get(key) != expected:
            issues.append(
                f"Audit model-setting mismatch for {key}: "
                f"{audit_model.get(key)!r} != {expected!r}."
            )

    status = _working_tree_status(repository_root)
    if require_clean_worktree and status:
        issues.append("Git working tree is not clean.")

    commit = _head_commit(repository_root)
    branch = _head_branch(repository_root)
    tags = _head_tags(repository_root)
    if require_head_tag and require_head_tag not in tags:
        issues.append(
            f"Required tag {require_head_tag!r} does not point at HEAD."
        )

    lock: dict[str, Any] = {
        "schema_version": "1.0",
        "lock_id": LAUNCH_LOCK_ID,
        "lock_version": LAUNCH_LOCK_VERSION,
        "status": "LOCKED" if not issues else "REFUSED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "plan_path": str(plan_path),
        "audit_path": str(audit_path),
        "intended_run_dir": str(intended_run_dir),
        "plan_sha256": plan["plan_sha256"],
        "audit_report_sha256": audit.get("audit_report_sha256"),
        "aggregate_prompt_sha256": audit.get(
            "aggregate_prompt_sha256"
        ),
        "aggregate_shared_candidate_sha256": audit.get(
            "aggregate_shared_candidate_sha256"
        ),
        "runner": expected_runner,
        "model_settings": expected_model,
        "pair_count": plan["pair_count"],
        "maximum_model_calls": plan["maximum_model_calls"],
        "git": {
            "commit_sha": commit,
            "branch": branch,
            "tags_at_head": tags,
            "working_tree_clean": not bool(status),
            "working_tree_status": status.splitlines(),
            "required_head_tag": require_head_tag,
        },
        "requirements": {
            "clean_worktree_required": require_clean_worktree,
            "head_tag_required": require_head_tag,
        },
        "issues": issues,
    }
    lock["launch_lock_sha256"] = _sha256_json(lock)
    return lock


def validate_launch_lock(
    lock: dict[str, Any],
    *,
    repository_root: Path,
    plan_path: Path,
    audit_path: Path,
    intended_run_dir: Path,
    model_name: str,
    max_output_tokens: int,
) -> list[str]:
    repository_root = repository_root.resolve()
    regenerated = create_launch_lock(
        plan_path=plan_path,
        audit_path=audit_path,
        repository_root=repository_root,
        intended_run_dir=intended_run_dir,
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        require_clean_worktree=bool(
            lock.get("requirements", {}).get(
                "clean_worktree_required", True
            )
        ),
        require_head_tag=lock.get("requirements", {}).get(
            "head_tag_required"
        ),
    )

    issues: list[str] = []
    stable_keys = [
        "status",
        "repository_root",
        "plan_path",
        "audit_path",
        "intended_run_dir",
        "plan_sha256",
        "audit_report_sha256",
        "aggregate_prompt_sha256",
        "aggregate_shared_candidate_sha256",
        "runner",
        "model_settings",
        "pair_count",
        "maximum_model_calls",
        "git",
        "requirements",
        "issues",
    ]
    for key in stable_keys:
        if lock.get(key) != regenerated.get(key):
            issues.append(f"Launch-lock mismatch for {key}.")
    return issues


def write_launch_lock(
    lock: dict[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(
            lock,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return path
