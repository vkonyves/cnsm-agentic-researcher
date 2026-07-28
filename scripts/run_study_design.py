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
from cnsm_agentic.study_design.tournament import (
    StudyDesignTournament,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate autonomous scientific study designs."
        )
    )
    parser.add_argument(
        "--programme",
        type=Path,
        required=True,
        help="Path to the research programme JSON configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for generated study-design artefacts.",
    )
    return parser.parse_args()


def load_programme(path: Path) -> ResearchProgramme:
    if not path.is_file():
        raise FileNotFoundError(
            f"Programme configuration does not exist: {path}"
        )

    try:
        raw_programme = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON programme configuration: {path}"
        ) from exc

    if not isinstance(raw_programme, dict):
        raise TypeError(
            "Programme configuration must contain a JSON object."
        )

    return ResearchProgramme.from_dict(raw_programme)


def write_common_tournament_outputs(
    output: Path,
    candidates: list[Any],
    tournament_result: dict[str, object],
) -> None:
    """
    Write artefacts that exist whether the tournament selects a winner
    or requires a finalist-resolution stage.
    """
    write_json(
        output / "candidates.json",
        [
            candidate.to_dict()
            for candidate in candidates
        ],
    )

    write_json(
        output / "critic_reviews.json",
        tournament_result["critic_reviews"],
    )

    write_json(
        output / "tournament.json",
        tournament_result,
    )


def build_manifest(
    *,
    programme_path: Path,
    output: Path,
    tournament_result: dict[str, object],
) -> dict[str, object]:
    return {
        "programme": str(programme_path),
        "programme_sha256": sha256_file(programme_path),
        "selection_status": tournament_result[
            "selection_status"
        ],
        "selected_candidate_id": tournament_result[
            "selected_candidate_id"
        ],
        "finalist_candidate_ids": tournament_result.get(
            "finalist_candidate_ids",
            [],
        ),
        "generated_files": sorted(
            path.name
            for path in output.glob("*.json")
        ),
    }


def handle_tie(
    *,
    args: argparse.Namespace,
    machine: ResearchStateMachine,
    tournament_result: dict[str, object],
) -> None:
    """
    Persist the tournament and stop cleanly when finalist resolution
    is required.

    The state remains CANDIDATES_FORMULATED because no final study
    design has yet been selected.
    """
    finalist_ids = tournament_result.get(
        "finalist_candidate_ids"
    )

    if not isinstance(finalist_ids, list) or not finalist_ids:
        raise ValueError(
            "Tie resolution was requested but no finalists were "
            "provided."
        )

    machine.add_gates(
        {
            "critic_reviews_complete",
            "selection_threshold_met",
            "tie_detected",
            "finalists_identified",
        }
    )

    write_json(
        args.output / "state.json",
        machine.to_dict(),
    )

    manifest = build_manifest(
        programme_path=args.programme,
        output=args.output,
        tournament_result=tournament_result,
    )

    write_json(
        args.output / "generation_manifest.json",
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
    print(
        "Tie resolution required. Finalists:",
        ", ".join(str(value) for value in finalist_ids),
    )
    print(
        "Research state remains:",
        machine.state.value,
    )


def handle_selection(
    *,
    args: argparse.Namespace,
    programme: ResearchProgramme,
    machine: ResearchStateMachine,
    candidates: list[Any],
    tournament_result: dict[str, object],
) -> None:
    selected_id = tournament_result.get(
        "selected_candidate_id"
    )

    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError(
            "Tournament reported a completed selection without a "
            "valid selected_candidate_id."
        )

    try:
        selected = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == selected_id
        )
    except StopIteration as exc:
        raise ValueError(
            f"Selected candidate {selected_id!r} does not exist "
            "in the generated candidate set."
        ) from exc

    machine.add_gates(
        {
            "critic_reviews_complete",
            "selection_threshold_met",
            "single_design_selected",
        }
    )

    machine.transition(
        ResearchState.DESIGN_SELECTED,
        {
            "critic_reviews_complete",
            "selection_threshold_met",
        },
    )

    plan = build_study_plan(
        selected,
        tournament_result,
    )

    write_json(
        args.output / "selected_study.json",
        asdict(plan),
    )

    write_json(
        args.output / "experiment_dag.json",
        [
            asdict(node)
            for node in plan.experiment_nodes
        ],
    )

    write_json(
        args.output / "claim_ledger.json",
        initialise_claim_ledger(plan),
    )

    write_json(
        args.output / "state.json",
        machine.to_dict(),
    )

    manifest = build_manifest(
        programme_path=args.programme,
        output=args.output,
        tournament_result=tournament_result,
    )

    write_json(
        args.output / "generation_manifest.json",
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
    print("Research state:", machine.state.value)


def main() -> None:
    args = parse_args()

    programme = load_programme(args.programme)
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    machine = ResearchStateMachine()

    machine.add_gates(
        {
            "mandate_validated",
        }
    )
    machine.transition(
        ResearchState.DISCOVERY_COMPLETE,
        {
            "mandate_validated",
        },
    )

    candidates = generate_seed_candidates(programme)

    machine.add_gates(
        {
            "candidate_schema_validated",
            "minimum_candidate_count_met",
        }
    )
    machine.transition(
        ResearchState.CANDIDATES_FORMULATED,
        {
            "candidate_schema_validated",
            "minimum_candidate_count_met",
        },
    )

    tournament_result = StudyDesignTournament(
        programme
    ).run(candidates)

    write_common_tournament_outputs(
        args.output,
        candidates,
        tournament_result,
    )

    selection_status = tournament_result.get(
        "selection_status"
    )

    if selection_status == "tie_resolution_required":
        handle_tie(
            args=args,
            machine=machine,
            tournament_result=tournament_result,
        )
        return

    if selection_status == "selected":
        handle_selection(
            args=args,
            programme=programme,
            machine=machine,
            candidates=candidates,
            tournament_result=tournament_result,
        )
        return

    raise ValueError(
        "Tournament returned an unsupported selection status: "
        f"{selection_status!r}"
    )


if __name__ == "__main__":
    main()