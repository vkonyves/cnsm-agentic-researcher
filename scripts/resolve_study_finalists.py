#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cnsm_agentic.study_design.candidates import (
    generate_seed_candidates,
)
from cnsm_agentic.study_design.claim_ledger import (
    initialise_claim_ledger,
)
from cnsm_agentic.study_design.finalist_resolution import (
    FinalistResolver,
)
from cnsm_agentic.study_design.models import ResearchProgramme
from cnsm_agentic.study_design.planner import build_study_plan
from cnsm_agentic.study_design.serialization import (
    sha256_file,
    write_json,
)
from cnsm_agentic.study_design.state_machine import (
    ResearchState,
    ResearchStateMachine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve tied autonomous study-design finalists."
        )
    )
    parser.add_argument(
        "--programme",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file does not exist: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main() -> None:
    args = parse_args()

    programme = ResearchProgramme.from_dict(
        read_json(args.programme)
    )

    tournament_path = (
        args.study_dir / "tournament.json"
    )
    tournament_before = read_json(tournament_path)

    if (
        tournament_before.get("selection_status")
        != "tie_resolution_required"
    ):
        raise ValueError(
            "Tournament is not awaiting finalist resolution"
        )

    finalist_ids = tournament_before.get(
        "finalist_candidate_ids"
    )

    if (
        not isinstance(finalist_ids, list)
        or len(finalist_ids) < 2
    ):
        raise ValueError(
            "Tournament does not contain at least two finalists"
        )

    candidates = generate_seed_candidates(programme)

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    unknown = [
        candidate_id
        for candidate_id in finalist_ids
        if candidate_id not in candidate_by_id
    ]

    if unknown:
        raise ValueError(
            f"Unknown finalist candidate IDs: {unknown}"
        )

    finalists = [
        candidate_by_id[candidate_id]
        for candidate_id in finalist_ids
    ]

    resolution = FinalistResolver(
        programme
    ).resolve(finalists)

    selected_id = str(
        resolution["selected_candidate_id"]
    )
    selected = candidate_by_id[selected_id]

    resolved_tournament = {
        **tournament_before,
        "selection_status": resolution[
            "selection_status"
        ],
        "selected_candidate_id": selected_id,
        "finalist_resolution": resolution,
        "selection_rationale": resolution[
            "selection_rationale"
        ],
    }

    plan = build_study_plan(
        selected,
        resolved_tournament,
    )

    machine = ResearchStateMachine(
        state=ResearchState.CANDIDATES_FORMULATED,
        satisfied_gates={
            "mandate_validated",
            "candidate_schema_validated",
            "minimum_candidate_count_met",
            "critic_reviews_complete",
            "selection_threshold_met",
            "tie_detected",
            "finalists_identified",
        },
    )

    machine.add_gates(
        {
            "finalist_reviews_complete",
            "finalist_resolution_complete",
            "single_design_selected",
        }
    )

    machine.transition(
        ResearchState.DESIGN_SELECTED,
        {
            "finalist_reviews_complete",
            "finalist_resolution_complete",
            "single_design_selected",
        },
    )

    write_json(
        args.study_dir / "finalist_resolution.json",
        resolution,
    )

    write_json(
        args.study_dir / "tournament.json",
        resolved_tournament,
    )

    write_json(
        args.study_dir / "selected_study.json",
        asdict(plan),
    )

    write_json(
        args.study_dir / "experiment_dag.json",
        [
            asdict(node)
            for node in plan.experiment_nodes
        ],
    )

    write_json(
        args.study_dir / "claim_ledger.json",
        initialise_claim_ledger(plan),
    )

    write_json(
        args.study_dir / "state.json",
        machine.to_dict(),
    )

    manifest = {
        "programme": str(args.programme),
        "programme_sha256": sha256_file(
            args.programme
        ),
        "selection_status": resolution[
            "selection_status"
        ],
        "selected_candidate_id": selected_id,
        "runner_up_candidate_id": resolution[
            "runner_up_candidate_id"
        ],
        "planned_follow_up_candidate_id": resolution[
            "planned_follow_up_candidate_id"
        ],
        "generated_files": sorted(
            path.name
            for path in args.study_dir.glob("*.json")
        ),
    }

    write_json(
        args.study_dir / "generation_manifest.json",
        manifest,
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("Selected candidate:", selected_id)
    print(
        "Planned follow-up:",
        resolution["planned_follow_up_candidate_id"],
    )
    print("Research state:", machine.state.value)


if __name__ == "__main__":
    main()
