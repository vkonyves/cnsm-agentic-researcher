#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cnsm_agentic.autonomous_research.final_guardrails import (
    assert_no_development_inputs,
    sha256_file,
)
from cnsm_agentic.autonomous_research.final_pipeline import (
    FinalAutonomousResearchPipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPOSITORY_ROOT / ".env"
EXPECTED_RUN_TYPE = "fresh_full_autonomous_research"


def load_environment() -> None:
    if ENV_PATH.is_file():
        load_dotenv(
            dotenv_path=ENV_PATH,
            override=False,
        )
        print("Loaded environment file:", ENV_PATH)
    else:
        print("No repository .env file found at:", ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not available. "
            f"Add it to {ENV_PATH} or export it."
        )

    if not os.getenv("OPENALEX_API_KEY"):
        raise RuntimeError(
            "OPENALEX_API_KEY is not available. "
            f"Add it to {ENV_PATH} or export it."
        )

    if not os.getenv("OPENALEX_MAILTO"):
        print("Warning: OPENALEX_MAILTO is not set.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a frozen autonomous research run."
    )
    parser.add_argument("--master-prompt", type=Path, required=True)
    parser.add_argument("--paper-run-constraints", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--development-rehearsal", action="store_true")
    parser.add_argument(
        "--execute-autonomous",
        action="store_true",
        help=(
            "Required acknowledgement that this command crosses "
            "the post-lock autonomy boundary."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def validate_hash(
    *, path: Path, expected_hash: str | None, description: str
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not expected_hash:
        raise ValueError(f"No expected hash recorded for {description}.")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{description} hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_framework_state(
    manifest: dict[str, Any], *, development_rehearsal: bool
) -> None:
    expected_commit = str(manifest.get("framework_commit", "")).strip()
    actual_commit = git("rev-parse", "HEAD")
    if not expected_commit:
        raise ValueError("Frozen framework commit is missing.")
    if actual_commit != expected_commit:
        raise ValueError(
            "Framework commit mismatch.\n"
            f"Frozen:  {expected_commit}\n"
            f"Current: {actual_commit}"
        )

    tracked_dirty = bool(
        git("status", "--porcelain", "--untracked-files=no")
    )
    if tracked_dirty and not development_rehearsal:
        raise ValueError(
            "Tracked Git worktree changed after the autonomy freeze."
        )

    frozen_tag = manifest.get("framework_tag")
    if frozen_tag:
        tags_at_head = set(git("tag", "--points-at", "HEAD").splitlines())
        if frozen_tag not in tags_at_head:
            raise ValueError(
                "Frozen framework tag is no longer present at HEAD: "
                f"{frozen_tag}"
            )


def validate_freeze_bundle(
    *,
    supplied_master_prompt: Path,
    supplied_constraints: Path,
    run_dir: Path,
    supplied_model: str,
    development_rehearsal: bool,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    provenance_dir = run_dir / "provenance"
    manifest_path = provenance_dir / "freeze_manifest.json"
    frozen_prompt_path = provenance_dir / "master_prompt.txt"
    frozen_policy_path = provenance_dir / "intervention_policy.json"
    frozen_capability_path = provenance_dir / "capability_manifest.json"
    frozen_constraints_path = provenance_dir / "paper_run_constraints.json"

    manifest = read_json(manifest_path)

    validate_hash(
        path=frozen_prompt_path,
        expected_hash=manifest.get("master_prompt_sha256"),
        description="Frozen master prompt",
    )
    validate_hash(
        path=frozen_policy_path,
        expected_hash=manifest.get("intervention_policy_sha256"),
        description="Frozen intervention policy",
    )
    validate_hash(
        path=frozen_capability_path,
        expected_hash=manifest.get("capability_manifest_sha256"),
        description="Frozen capability manifest",
    )
    validate_hash(
        path=frozen_constraints_path,
        expected_hash=manifest.get("paper_run_constraints_sha256"),
        description="Frozen paper-run constraints",
    )

    for supplied, frozen, description in (
        (supplied_master_prompt, frozen_prompt_path, "master prompt"),
        (supplied_constraints, frozen_constraints_path, "paper-run constraints"),
    ):
        if not supplied.is_file():
            raise FileNotFoundError(
                f"Supplied {description} not found: {supplied}"
            )
        supplied_hash = sha256_file(supplied)
        frozen_hash = sha256_file(frozen)
        if supplied_hash != frozen_hash:
            raise ValueError(
                f"Supplied {description} does not match frozen {description}.\n"
                f"Supplied: {supplied_hash}\n"
                f"Frozen:   {frozen_hash}"
            )

    frozen_rehearsal_mode = bool(
        manifest.get("development_rehearsal", False)
    )
    if frozen_rehearsal_mode != development_rehearsal:
        raise ValueError(
            "Run/freeze mode mismatch. "
            f"Frozen development_rehearsal={frozen_rehearsal_mode}, "
            f"launcher development_rehearsal={development_rehearsal}."
        )

    frozen_model = str(manifest.get("model") or "").strip()
    if not frozen_model:
        raise ValueError("Frozen model is missing from freeze manifest.")
    if supplied_model != frozen_model:
        raise ValueError(
            "Model mismatch.\n"
            f"Frozen:   {frozen_model}\n"
            f"Supplied: {supplied_model}"
        )

    frozen_run_type = str(manifest.get("run_type") or "").strip()
    if frozen_run_type != EXPECTED_RUN_TYPE:
        raise ValueError(
            "Unexpected frozen run type.\n"
            f"Expected: {EXPECTED_RUN_TYPE}\n"
            f"Frozen:   {frozen_run_type}"
        )

    constraints = read_json(frozen_constraints_path)
    if constraints.get("run_type") != EXPECTED_RUN_TYPE:
        raise ValueError(
            "Paper-run constraints do not declare the required "
            "fresh autonomous run type."
        )
    if (
        constraints.get("prelock_development_results_as_final_evidence")
        is not False
    ):
        raise ValueError(
            "Paper-run constraints must explicitly forbid pre-lock "
            "development results as final evidence."
        )

    fmt = constraints.get("format") or {}
    if (
        fmt.get("maximum_pages") != 5
        or fmt.get("references_included_in_limit") is not True
        or fmt.get("disclosure_statement_included_in_limit") is not True
        or fmt.get("disclosure_statement_must_be_within_pages_1_to_5")
        is not True
    ):
        raise ValueError(
            "Frozen paper-run constraints do not enforce the required "
            "five-page limit including references and Disclosure Statement."
        )

    validate_framework_state(
        manifest,
        development_rehearsal=development_rehearsal,
    )

    return (
        manifest,
        frozen_prompt_path,
        frozen_policy_path,
        frozen_capability_path,
        frozen_constraints_path,
    )


async def run_pipeline(
    *,
    master_prompt_path: Path,
    run_dir: Path,
    model: str,
    development_rehearsal: bool,
    capability_manifest: dict[str, Any],
    paper_run_constraints: dict[str, Any],
) -> None:
    pipeline = FinalAutonomousResearchPipeline(
        model=model,
        development_rehearsal=development_rehearsal,
    )
    result = await pipeline.run(
        master_prompt=master_prompt_path.read_text(encoding="utf-8"),
        run_dir=run_dir,
        capability_manifest=capability_manifest,
        paper_run_constraints=paper_run_constraints,
    )
    print("Final autonomous research run complete")
    print("Ready:", result.ready)
    print("State:", result.final_state)


def main() -> None:
    args = parse_args()

    if not args.execute_autonomous:
        raise SystemExit(
            "Refusing to cross the autonomy boundary without "
            "--execute-autonomous."
        )

    (
        manifest,
        frozen_prompt_path,
        _frozen_policy_path,
        frozen_capability_path,
        frozen_constraints_path,
    ) = validate_freeze_bundle(
        supplied_master_prompt=args.master_prompt,
        supplied_constraints=args.paper_run_constraints,
        run_dir=args.run_dir,
        supplied_model=args.model,
        development_rehearsal=args.development_rehearsal,
    )

    capability_manifest = read_json(frozen_capability_path)
    paper_run_constraints = read_json(
        frozen_constraints_path
    )

    assert_no_development_inputs(
        {
            "master_prompt": frozen_prompt_path.read_text(encoding="utf-8"),
            "capability_manifest": capability_manifest,
            "paper_run_constraints": paper_run_constraints,
        }
    )

    load_environment()

    print("AUTONOMY GATE: PASS")
    print("Framework commit:", manifest.get("framework_commit"))
    print("Framework tag:", manifest.get("framework_tag"))
    print("Frozen model:", manifest.get("model"))
    print("Run type:", manifest.get("run_type"))
    print("Capability manifest:", frozen_capability_path)
    print("Paper-run constraints:", frozen_constraints_path)
    print("Crossing post-lock autonomy boundary: YES")

    asyncio.run(
        run_pipeline(
            master_prompt_path=frozen_prompt_path,
            run_dir=args.run_dir,
            model=args.model,
            development_rehearsal=args.development_rehearsal,
            capability_manifest=capability_manifest,
            paper_run_constraints=paper_run_constraints,
        )
    )


if __name__ == "__main__":
    main()
