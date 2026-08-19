from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .final_guardrails import (
    sha256_file,
)
from .final_schemas import (
    FrozenRunManifest,
)


def git(
    *args: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def create_freeze_bundle(
    *,
    master_prompt: Path,
    intervention_policy: Path,
    capability_manifest: Path,
    output_dir: Path,
    development_rehearsal: bool,
    paper_run_constraints: Path | None = None,
    model: str | None = None,
    run_type: str | None = None,
) -> FrozenRunManifest:
    if not master_prompt.is_file():
        raise FileNotFoundError(
            f"Master prompt not found: {master_prompt}"
        )

    if not intervention_policy.is_file():
        raise FileNotFoundError(
            "Intervention policy not found: "
            f"{intervention_policy}"
        )

    if not capability_manifest.is_file():
        raise FileNotFoundError(
            "Capability manifest not found: "
            f"{capability_manifest}"
        )

    if (
        paper_run_constraints is not None
        and not paper_run_constraints.is_file()
    ):
        raise FileNotFoundError(
            "Paper-run constraints not found: "
            f"{paper_run_constraints}"
        )

    if model is not None and not model.strip():
        raise ValueError("Frozen model must be non-empty.")

    if run_type is not None and not run_type.strip():
        raise ValueError("Frozen run type must be non-empty.")

    commit = git(
        "rev-parse",
        "HEAD",
    )

    dirty = bool(
        git(
            "status",
            "--porcelain",
        )
    )

    if (
        dirty
        and not development_rehearsal
    ):
        raise ValueError(
            "Clean worktree required for final freeze"
        )

    tags = git(
        "tag",
        "--points-at",
        "HEAD",
    ).splitlines()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_out = (
        output_dir / "master_prompt.txt"
    )
    policy_out = (
        output_dir
        / "intervention_policy.json"
    )
    capability_out = (
        output_dir
        / "capability_manifest.json"
    )
    constraints_out = (
        output_dir
        / "paper_run_constraints.json"
    )

    prompt_out.write_bytes(
        master_prompt.read_bytes()
    )
    policy_out.write_bytes(
        intervention_policy.read_bytes()
    )
    capability_out.write_bytes(
        capability_manifest.read_bytes()
    )

    constraints_hash: str | None = None
    if paper_run_constraints is not None:
        constraints_out.write_bytes(
            paper_run_constraints.read_bytes()
        )
        constraints_hash = sha256_file(
            constraints_out
        )

    manifest = FrozenRunManifest(
        schema_version=(
            "1.1"
            if paper_run_constraints is not None
            else "1.0"
        ),
        framework_commit=commit,
        framework_tag=(
            tags[0]
            if tags
            else None
        ),
        framework_dirty=dirty,
        master_prompt_sha256=(
            sha256_file(
                prompt_out
            )
        ),
        intervention_policy_sha256=(
            sha256_file(
                policy_out
            )
        ),
        capability_manifest_sha256=(
            sha256_file(
                capability_out
            )
        ),
        paper_run_constraints_sha256=(
            constraints_hash
        ),
        model=(
            model.strip()
            if model is not None
            else None
        ),
        run_type=(
            run_type.strip()
            if run_type is not None
            else None
        ),
        created_at_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        development_rehearsal=(
            development_rehearsal
        ),
    )

    (
        output_dir
        / "framework_commit.txt"
    ).write_text(
        commit + "\n",
        encoding="utf-8",
    )

    for filename, value in (
        ("master_prompt.sha256", manifest.master_prompt_sha256),
        ("intervention_policy.sha256", manifest.intervention_policy_sha256),
        ("capability_manifest.sha256", manifest.capability_manifest_sha256),
    ):
        (output_dir / filename).write_text(
            value + "\n",
            encoding="utf-8",
        )

    if manifest.paper_run_constraints_sha256:
        (
            output_dir
            / "paper_run_constraints.sha256"
        ).write_text(
            manifest.paper_run_constraints_sha256 + "\n",
            encoding="utf-8",
        )

    (
        output_dir
        / "freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "intervention_log.jsonl"
    ).touch(
        exist_ok=True,
    )

    return manifest
