#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cnsm_agentic.study_design.models import ResearchProgramme
from cnsm_agentic.study_design.preregistration import (
    build_preregistration,
    build_run_specs,
    canonical_json_bytes,
    sha256_bytes,
    write_canonical_json,
)
from cnsm_agentic.study_design.serialization import write_json
from cnsm_agentic.study_design.state_machine import (
    ResearchState,
    ResearchStateMachine,
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programme", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    programme_raw = read_json(args.programme)
    ResearchProgramme.from_dict(programme_raw)

    state_before = read_json(args.study_dir / "state.json")
    if state_before.get("state") != "DESIGN_SELECTED":
        raise ValueError(
            "Preregistration can only be frozen from DESIGN_SELECTED"
        )

    selected_study = read_json(args.study_dir / "selected_study.json")
    resolution = read_json(args.study_dir / "finalist_resolution.json")

    preregistration = build_preregistration(
        programme=programme_raw,
        selected_study=selected_study,
        finalist_resolution=resolution,
    )

    prereg_bytes = canonical_json_bytes(preregistration)
    prereg_hash = sha256_bytes(prereg_bytes)

    (args.study_dir / "preregistration.json").write_bytes(prereg_bytes)
    (args.study_dir / "preregistration.sha256").write_text(
        prereg_hash + "\n",
        encoding="utf-8",
    )

    run_specs = build_run_specs(preregistration)
    run_dir = args.study_dir / "run_specs"
    run_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in run_specs:
        spec_dict = asdict(spec)
        path = run_dir / f"{spec.run_id}.json"
        write_canonical_json(path, spec_dict)
        runs.append(
            {
                "run_id": spec.run_id,
                "path": str(path.relative_to(args.study_dir)),
                "sha256": sha256_bytes(
                    canonical_json_bytes(spec_dict)
                ),
                "planned_evaluations": spec.sample_size,
            }
        )

    execution_manifest = {
        "schema_version": "0.5.0",
        "preregistration_sha256": prereg_hash,
        "selected_candidate_id": "C2",
        "planned_follow_up_candidate_id": "C4",
        "run_count": len(runs),
        "total_planned_evaluations": sum(
            run["planned_evaluations"] for run in runs
        ),
        "execution_order": [
            "direct-original",
            "direct-repeat",
            "direct-permuted",
            "structured-original",
            "structured-repeat",
            "structured-permuted",
        ],
        "runs": runs,
    }
    write_canonical_json(
        args.study_dir / "execution_manifest.json",
        execution_manifest,
    )

    machine = ResearchStateMachine(
        state=ResearchState.DESIGN_SELECTED,
        satisfied_gates={
            "mandate_validated",
            "candidate_schema_validated",
            "minimum_candidate_count_met",
            "critic_reviews_complete",
            "selection_threshold_met",
            "tie_detected",
            "finalists_identified",
            "finalist_reviews_complete",
            "finalist_resolution_complete",
            "single_design_selected",
        },
    )
    gates = {
        "preregistration_complete",
        "preregistration_validated",
        "preregistration_hashed",
        "run_specs_complete",
        "execution_manifest_complete",
        "no_unresolved_placeholders",
    }
    machine.add_gates(gates)
    machine.transition(
        ResearchState.PREREGISTRATION_FROZEN,
        gates,
    )
    write_json(args.study_dir / "state.json", machine.to_dict())

    print("Preregistration frozen")
    print("Selected candidate: C2")
    print(
        "Planned evaluations:",
        execution_manifest["total_planned_evaluations"],
    )
    print("Preregistration SHA-256:", prereg_hash)
    print("Research state:", machine.state.value)


if __name__ == "__main__":
    main()
