#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from cnsm_agentic.autonomous_research import (
    AutonomousDiscoveryPipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--programme",
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
        "--per-source-per-query",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-synthesis-records",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    programme = json.loads(
        args.programme.read_text(
            encoding="utf-8"
        )
    )

    pipeline = AutonomousDiscoveryPipeline(
        model=args.model,
        per_source_per_query=args.per_source_per_query,
        max_synthesis_records=(
            args.max_synthesis_records
        ),
    )

    decision = asyncio.run(
        pipeline.run(
            programme=programme,
            run_dir=args.run_dir,
        )
    )

    print("Autonomous discovery complete")
    print(
        "Selected candidate:",
        decision.selected_candidate_id,
    )
    print(
        "State: AUTONOMOUS_DESIGN_SELECTED"
    )


if __name__ == "__main__":
    main()