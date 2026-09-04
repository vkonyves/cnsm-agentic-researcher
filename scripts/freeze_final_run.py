#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cnsm_agentic.autonomous_research.freeze import (
    create_freeze_bundle,
)

DEFAULT_RUN_TYPE = "fresh_full_autonomous_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze framework, prompt, intervention policy, capability "
            "manifest, paper-run constraints, and model."
        )
    )
    parser.add_argument("--master-prompt", type=Path, required=True)
    parser.add_argument("--intervention-policy", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--paper-run-constraints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-type", default=DEFAULT_RUN_TYPE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-rehearsal", action="store_true")
    return parser.parse_args()


def get_manifest_value(manifest: Any, field_name: str) -> Any:
    if isinstance(manifest, dict):
        return manifest[field_name]
    return getattr(manifest, field_name)


def main() -> None:
    args = parse_args()
    for description, path in {
        "Master prompt": args.master_prompt,
        "Intervention policy": args.intervention_policy,
        "Capability manifest": args.capability_manifest,
        "Paper-run constraints": args.paper_run_constraints,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    manifest = create_freeze_bundle(
        master_prompt=args.master_prompt,
        intervention_policy=args.intervention_policy,
        capability_manifest=args.capability_manifest,
        paper_run_constraints=args.paper_run_constraints,
        model=args.model,
        run_type=args.run_type,
        output_dir=args.output_dir / "provenance",
        development_rehearsal=args.development_rehearsal,
    )

    print("Freeze bundle created")
    print("Output directory:", args.output_dir)
    for label, field in (
        ("Framework commit", "framework_commit"),
        ("Framework tag", "framework_tag"),
        ("Master prompt SHA-256", "master_prompt_sha256"),
        ("Intervention policy SHA-256", "intervention_policy_sha256"),
        ("Capability manifest SHA-256", "capability_manifest_sha256"),
        ("Paper-run constraints SHA-256", "paper_run_constraints_sha256"),
        ("Frozen model", "model"),
        ("Run type", "run_type"),
        ("Development rehearsal", "development_rehearsal"),
    ):
        print(f"{label}:", get_manifest_value(manifest, field))


if __name__ == "__main__":
    main()
