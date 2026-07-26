from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnsm_agentic.inspection import (
    build_inspection_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform read-only structural inspection "
            "of acquired benchmark assets."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Completed research-pilot run directory.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    inventory = build_inspection_inventory(
        source_run=args.source_run
    )

    summary = {
        "inspected_count": (
            inventory.inspected_count
        ),
        "unsupported_count": (
            inventory.unsupported_count
        ),
        "failed_count": (
            inventory.failed_count
        ),
        "unsafe_zip_member_count": (
            inventory.unsafe_zip_member_count
        ),
        "suspicious_zip_member_count": (
            inventory.suspicious_zip_member_count
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nSaved inventory:",
        args.source_run
        / "inspection"
        / "inspection_inventory.json",
    )

    return (
        0
        if inventory.failed_count == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
