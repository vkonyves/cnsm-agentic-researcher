#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from cnsm_agentic.autonomous_research import (
    AutonomousDiscoveryPipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPOSITORY_ROOT / ".env"


def load_environment() -> None:
    """
    Load environment variables from the repository-root .env file.

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

    openalex_api_key = os.getenv(
        "OPENALEX_API_KEY"
    )

    if not openalex_api_key:
        raise RuntimeError(
            "OPENALEX_API_KEY is not available. "
            f"Add it to {ENV_PATH} or export it in the shell."
        )

    if not os.getenv("OPENALEX_MAILTO"):
        print(
            "Warning: OPENALEX_MAILTO is not set. "
            "OpenAlex requests will run without a mailto identifier."
        )


def main() -> None:
    load_environment()

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

    if not args.programme.is_file():
        raise FileNotFoundError(
            f"Programme file not found: {args.programme}"
        )

    programme = json.loads(
        args.programme.read_text(
            encoding="utf-8",
        )
    )

    pipeline = AutonomousDiscoveryPipeline(
        model=args.model,
        per_source_per_query=(
            args.per_source_per_query
        ),
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

    print(
        "Autonomous discovery complete"
    )
    print(
        "Selected candidate:",
        decision.selected_candidate_id,
    )
    print(
        "State: AUTONOMOUS_DESIGN_SELECTED"
    )


if __name__ == "__main__":
    main()