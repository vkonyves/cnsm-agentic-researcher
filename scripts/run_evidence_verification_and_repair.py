#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cnsm_agentic.autonomous_research.design_repair import (
    DESIGN_REPAIR_AGENT,
    READINESS_JUDGE,
    run_agent_with_retry,
)
from cnsm_agentic.autonomous_research.evidence_verification import (
    verify_evidence,
)
from cnsm_agentic.autonomous_research.repair_schemas import (
    RepairedStudyDesign,
    RepairReadinessReport,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPOSITORY_ROOT / ".env"


def load_environment() -> None:
    """
    Load repository-local environment variables.

    Expected values may include:
    - OPENAI_API_KEY
    - OPENALEX_MAILTO
    """
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
        print(
            "Using variables already present in the shell environment."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not available. "
            f"Add it to {ENV_PATH} or export it in the shell."
        )

    if not os.getenv("OPENALEX_MAILTO"):
        print(
            "Warning: OPENALEX_MAILTO is not set. "
            "This script does not normally query OpenAlex directly, "
            "but the value may be needed by reused literature components."
        )


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(value, "model_dump"):
        value = value.model_dump()

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


async def run(
    args: argparse.Namespace,
) -> None:
    programme = read_json(
        args.programme
    )

    discovery_run = (
        args.discovery_run
    )

    paths = {
        "records": (
            discovery_run
            / "literature/records.json"
        ),
        "synthesis": (
            discovery_run
            / "literature/evidence_synthesis.json"
        ),
        "candidates": (
            discovery_run
            / "selection/candidates.json"
        ),
        "reviews": (
            discovery_run
            / "selection/critic_reviews.json"
        ),
        "decision": (
            discovery_run
            / "selection/decision.json"
        ),
        "validation": (
            discovery_run
            / "selection/candidate_validation.json"
        ),
    }

    data = {
        name: read_json(path)
        for name, path in paths.items()
    }

    validation_status = (
        data["validation"].get(
            "candidate_validation_status"
        )
    )

    if validation_status != "passed":
        raise ValueError(
            "v0.6 candidate validation failed. "
            f"Received status: {validation_status!r}"
        )

    selected_candidate_id = (
        data["decision"][
            "selected_candidate_id"
        ]
    )

    candidates_by_id = {
        candidate["candidate_id"]: (
            candidate
        )
        for candidate in (
            data["candidates"][
                "candidates"
            ]
        )
    }

    if (
        selected_candidate_id
        not in candidates_by_id
    ):
        raise ValueError(
            "Selection decision references an "
            "unknown candidate: "
            f"{selected_candidate_id}"
        )

    evidence_verification = (
        verify_evidence(
            records=data["records"],
            synthesis=data["synthesis"],
            candidates=data["candidates"],
            decision=data["decision"],
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        args.output_dir
        / "evidence_verification.json",
        evidence_verification.to_dict(),
    )

    repair_payload = {
        "programme": programme,
        "selected_candidate": (
            candidates_by_id[
                selected_candidate_id
            ]
        ),
        "selection_decision": (
            data["decision"]
        ),
        "critic_reviews": (
            data["reviews"]
        ),
        "evidence_synthesis": (
            data["synthesis"]
        ),
        "evidence_verification": (
            evidence_verification.to_dict()
        ),
        "allowed_evidence_record_ids": sorted(
            record["record_id"]
            for record in data["records"]
        ),
    }

    repaired_design = (
        await run_agent_with_retry(
            DESIGN_REPAIR_AGENT,
            repair_payload,
            expected_type=(
                RepairedStudyDesign
            ),
            stage_name=(
                "Autonomous design repair"
            ),
        )
    )

    if (
        repaired_design.selected_candidate_id
        != selected_candidate_id
    ):
        raise ValueError(
            "Repaired design candidate mismatch. "
            f"Expected {selected_candidate_id}, "
            f"received "
            f"{repaired_design.selected_candidate_id}."
        )

    allowed_evidence_ids = {
        record["record_id"]
        for record in data["records"]
    }

    unknown_evidence_ids = sorted(
        set(
            repaired_design.evidence_record_ids
        )
        - allowed_evidence_ids
    )

    if unknown_evidence_ids:
        raise ValueError(
            "Repaired design references unknown "
            "evidence IDs: "
            f"{unknown_evidence_ids}"
        )

    write_json(
        args.output_dir
        / "repaired_design.json",
        repaired_design,
    )

    readiness_payload = {
        "programme": programme,
        "selected_candidate_id": (
            selected_candidate_id
        ),
        "evidence_verification": (
            evidence_verification.to_dict()
        ),
        "repaired_design": (
            repaired_design.model_dump()
        ),
    }

    readiness = (
        await run_agent_with_retry(
            READINESS_JUDGE,
            readiness_payload,
            expected_type=(
                RepairReadinessReport
            ),
            stage_name=(
                "Readiness judgement"
            ),
        )
    )

    if (
        readiness.selected_candidate_id
        != selected_candidate_id
    ):
        raise ValueError(
            "Readiness report candidate mismatch. "
            f"Expected {selected_candidate_id}, "
            f"received "
            f"{readiness.selected_candidate_id}."
        )

    is_ready = (
        not evidence_verification.critical_issues
        and (
            repaired_design
            .preregistration_fields_complete
        )
        and not (
            repaired_design
            .unresolved_critical_issues
        )
    )

    readiness.next_state = (
        "FRAMEWORK_VALIDATED_FOR_FINAL_RUN"
        if is_ready
        else "DESIGN_REPAIR_REQUIRED"
    )

    write_json(
        args.output_dir
        / "readiness_report.json",
        readiness,
    )

    write_json(
        args.output_dir
        / "provenance.json",
        {
            "schema_version": "0.7.0",
            "programme_path": str(
                args.programme
            ),
            "programme_sha256": (
                sha256_file(
                    args.programme
                )
            ),
            "discovery_run": str(
                args.discovery_run
            ),
            "source_hashes": {
                name: sha256_file(path)
                for name, path
                in paths.items()
            },
            "selected_candidate_id": (
                selected_candidate_id
            ),
            "evidence_quality_score": (
                evidence_verification
                .quality_score
            ),
            "next_state": (
                readiness.next_state
            ),
            "development_rehearsal": True,
        },
    )

    write_json(
        args.output_dir
        / "state.json",
        {
            "state": (
                readiness.next_state
            ),
            "selected_candidate_id": (
                selected_candidate_id
            ),
            "development_rehearsal": True,
        },
    )

    print(
        "Evidence verification and design repair complete"
    )
    print(
        "Selected candidate:",
        selected_candidate_id,
    )
    print(
        "Evidence quality score:",
        evidence_verification.quality_score,
    )
    print(
        "Next state:",
        readiness.next_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify evidence and autonomously "
            "repair a selected study design."
        )
    )

    parser.add_argument(
        "--programme",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--discovery-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
    )

    return parser.parse_args()


def main() -> None:
    load_environment()

    args = parse_args()

    if not args.programme.is_file():
        raise FileNotFoundError(
            f"Programme file not found: "
            f"{args.programme}"
        )

    if not args.discovery_run.is_dir():
        raise NotADirectoryError(
            "Discovery run directory not found: "
            f"{args.discovery_run}"
        )

    DESIGN_REPAIR_AGENT.model = (
        args.model
    )
    READINESS_JUDGE.model = (
        args.model
    )

    asyncio.run(
        run(args)
    )


if __name__ == "__main__":
    main()