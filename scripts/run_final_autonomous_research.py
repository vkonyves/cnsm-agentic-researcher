#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cnsm_agentic.autonomous_research.final_guardrails import (
    sha256_file,
)
from cnsm_agentic.autonomous_research.final_pipeline import (
    FinalAutonomousResearchPipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPOSITORY_ROOT / ".env"


def load_environment() -> None:
    if ENV_PATH.is_file():
        load_dotenv(
            dotenv_path=ENV_PATH,
            override=False,
        )
        print(
            "Loaded environment file:",
            ENV_PATH,
        )
    else:
        print(
            "No repository .env file found at:",
            ENV_PATH,
        )

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
        print(
            "Warning: OPENALEX_MAILTO is not set."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a frozen autonomous research run."
        )
    )

    parser.add_argument(
        "--master-prompt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
    )
    parser.add_argument(
        "--development-rehearsal",
        action="store_true",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in {path}"
        )

    return value


def validate_hash(
    *,
    path: Path,
    expected_hash: str | None,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    if not expected_hash:
        raise ValueError(
            f"No expected hash recorded for {description}."
        )

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"{description} hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )


def validate_freeze_bundle(
    *,
    supplied_master_prompt: Path,
    run_dir: Path,
    development_rehearsal: bool,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    provenance_dir = (
        run_dir / "provenance"
    )

    manifest_path = (
        provenance_dir
        / "freeze_manifest.json"
    )
    frozen_prompt_path = (
        provenance_dir
        / "master_prompt.txt"
    )
    frozen_policy_path = (
        provenance_dir
        / "intervention_policy.json"
    )
    frozen_capability_path = (
        provenance_dir
        / "capability_manifest.json"
    )

    manifest = read_json(
        manifest_path
    )

    validate_hash(
        path=frozen_prompt_path,
        expected_hash=manifest.get(
            "master_prompt_sha256"
        ),
        description="Frozen master prompt",
    )

    validate_hash(
        path=frozen_policy_path,
        expected_hash=manifest.get(
            "intervention_policy_sha256"
        ),
        description="Frozen intervention policy",
    )

    validate_hash(
        path=frozen_capability_path,
        expected_hash=manifest.get(
            "capability_manifest_sha256"
        ),
        description="Frozen capability manifest",
    )

    if not supplied_master_prompt.is_file():
        raise FileNotFoundError(
            "Supplied master prompt not found: "
            f"{supplied_master_prompt}"
        )

    supplied_prompt_hash = sha256_file(
        supplied_master_prompt
    )

    frozen_prompt_hash = sha256_file(
        frozen_prompt_path
    )

    if (
        supplied_prompt_hash
        != frozen_prompt_hash
    ):
        raise ValueError(
            "Supplied master prompt does not match "
            "the frozen master prompt.\n"
            f"Supplied: {supplied_prompt_hash}\n"
            f"Frozen:   {frozen_prompt_hash}"
        )

    frozen_rehearsal_mode = bool(
        manifest.get(
            "development_rehearsal",
            False,
        )
    )

    if (
        frozen_rehearsal_mode
        != development_rehearsal
    ):
        raise ValueError(
            "Run/freeze mode mismatch. "
            f"Frozen development_rehearsal="
            f"{frozen_rehearsal_mode}, "
            f"launcher development_rehearsal="
            f"{development_rehearsal}."
        )

    return (
        manifest,
        frozen_prompt_path,
        frozen_policy_path,
        frozen_capability_path,
    )


async def run_pipeline(
    *,
    master_prompt_path: Path,
    run_dir: Path,
    model: str,
    development_rehearsal: bool,
    capability_manifest: dict[str, Any],
) -> None:
    pipeline = FinalAutonomousResearchPipeline(
        model=model,
        development_rehearsal=(
            development_rehearsal
        ),
    )

    result = await pipeline.run(
        master_prompt=(
            master_prompt_path.read_text(
                encoding="utf-8",
            )
        ),
        run_dir=run_dir,
        capability_manifest=(
            capability_manifest
        ),
    )

    print(
        "Final autonomous bootstrap run complete"
    )
    print(
        "Ready:",
        result.ready,
    )
    print(
        "State:",
        result.final_state,
    )


def main() -> None:
    load_environment()

    args = parse_args()

    (
        manifest,
        frozen_prompt_path,
        _frozen_policy_path,
        frozen_capability_path,
    ) = validate_freeze_bundle(
        supplied_master_prompt=(
            args.master_prompt
        ),
        run_dir=args.run_dir,
        development_rehearsal=(
            args.development_rehearsal
        ),
    )

    capability_manifest = read_json(
        frozen_capability_path
    )

    print(
        "Framework commit:",
        manifest.get(
            "framework_commit"
        ),
    )
    print(
        "Capability manifest:",
        frozen_capability_path,
    )

    asyncio.run(
        run_pipeline(
            master_prompt_path=(
                frozen_prompt_path
            ),
            run_dir=args.run_dir,
            model=args.model,
            development_rehearsal=(
                args.development_rehearsal
            ),
            capability_manifest=(
                capability_manifest
            ),
        )
    )


if __name__ == "__main__":
    main()