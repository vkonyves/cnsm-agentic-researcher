from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from cnsm_agentic.acquisition import (
    execute_acquisition_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and hash assets from a "
            "deterministic acquisition plan."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help=(
            "Path to the completed research-pilot run."
        ),
    )

    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()

    manifest = await execute_acquisition_plan(
        source_run=args.source_run,
    )

    print(
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nDownloaded:",
        manifest.downloaded_count,
    )

    print(
        "Skipped:",
        manifest.skipped_count,
    )

    print(
        "Failed:",
        manifest.failed_count,
    )

    return (
        0
        if manifest.failed_count == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main_async()
        )
    )
