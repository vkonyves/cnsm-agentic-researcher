from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnsm_agentic.acquisition import build_acquisition_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic acquisition plan from a "
            "completed research-pilot run."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Path to the completed research-pilot run.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan = build_acquisition_plan(
        source_run=args.source_run,
    )

    output_directory = (
        args.source_run
        / "acquisition"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "acquisition_plan.json"
    )

    output_path.write_text(
        plan.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            plan.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nSaved acquisition plan: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
