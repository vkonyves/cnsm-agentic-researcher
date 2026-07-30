#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cnsm_agentic.autonomous_research.freeze import (
    create_freeze_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the framework, master prompt, intervention policy, "
            "and capability manifest for an autonomous research run."
        )
    )

    parser.add_argument(
        "--master-prompt",
        type=Path,
        required=True,
        help="Path to the Research Director master prompt.",
    )

    parser.add_argument(
        "--intervention-policy",
        type=Path,
        default=Path(
            "configs/final_run/intervention_policy.json"
        ),
        help="Path to the human-intervention policy.",
    )

    parser.add_argument(
        "--capability-manifest",
        type=Path,
        default=Path(
            "configs/final_run/capability_manifest.json"
        ),
        help=(
            "Path to the execution capability manifest "
            "describing the available infrastructure."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to create the frozen provenance bundle.",
    )

    parser.add_argument(
        "--development-rehearsal",
        action="store_true",
        help=(
            "Allow a development rehearsal freeze. "
            "A real final freeze requires a clean Git worktree."
        ),
    )

    return parser.parse_args()


def get_manifest_value(
    manifest: Any,
    field_name: str,
) -> Any:
    """
    Support either a dictionary or a Pydantic model returned
    by create_freeze_bundle().
    """
    if isinstance(manifest, dict):
        return manifest[field_name]

    return getattr(
        manifest,
        field_name,
    )


def validate_inputs(
    args: argparse.Namespace,
) -> None:
    required_files = {
        "Master prompt": args.master_prompt,
        "Intervention policy": (
            args.intervention_policy
        ),
        "Capability manifest": (
            args.capability_manifest
        ),
    }

    for description, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )


def main() -> None:
    args = parse_args()

    validate_inputs(args)

    manifest = create_freeze_bundle(
        master_prompt=args.master_prompt,
        intervention_policy=(
            args.intervention_policy
        ),
        capability_manifest=(
            args.capability_manifest
        ),
        output_dir=args.output_dir,
        development_rehearsal=(
            args.development_rehearsal
        ),
    )

    framework_commit = get_manifest_value(
        manifest,
        "framework_commit",
    )
    master_prompt_sha256 = get_manifest_value(
        manifest,
        "master_prompt_sha256",
    )
    intervention_policy_sha256 = (
        get_manifest_value(
            manifest,
            "intervention_policy_sha256",
        )
    )
    capability_manifest_sha256 = (
        get_manifest_value(
            manifest,
            "capability_manifest_sha256",
        )
    )
    development_rehearsal = (
        get_manifest_value(
            manifest,
            "development_rehearsal",
        )
    )

    print("Freeze bundle created")
    print(
        "Output directory:",
        args.output_dir,
    )
    print(
        "Framework commit:",
        framework_commit,
    )
    print(
        "Master prompt SHA-256:",
        master_prompt_sha256,
    )
    print(
        "Intervention policy SHA-256:",
        intervention_policy_sha256,
    )
    print(
        "Capability manifest SHA-256:",
        capability_manifest_sha256,
    )
    print(
        "Development rehearsal:",
        development_rehearsal,
    )


if __name__ == "__main__":
    main()