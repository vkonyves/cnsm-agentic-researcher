from __future__ import annotations

import argparse
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_plan_subset import (
    create_subset_from_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a frozen subset of a controlled-fault plan."
    )
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument(
        "--pair-ids",
        required=True,
        help="Comma-separated frozen pair IDs, e.g. pair-000002,pair-000008.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pair_ids = [
        item.strip()
        for item in args.pair_ids.split(",")
        if item.strip()
    ]
    subset = create_subset_from_file(
        args.parent_plan,
        pair_ids=pair_ids,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    print()
    print("CONTROLLED-FAULT SUBSET PLAN")
    print("============================")
    print(f"Output:                  {args.output.resolve()}")
    print(f"Parent plan SHA-256:     {subset['parent_plan_sha256']}")
    print(f"Subset plan SHA-256:     {subset['plan_sha256']}")
    print(f"Selected pairs:          {','.join(subset['selected_pair_ids'])}")
    print(f"Pair count:              {subset['pair_count']}")
    print(f"Maximum model calls:     {subset['maximum_model_calls']}")
    print("Assignments:")
    for pair in subset["pairs"]:
        print(
            f"  {pair['pair_id']}: "
            f"task={pair['task_index']}, "
            f"pattern={pair['workflow_pattern']}, "
            f"fault={pair['fault_class']}"
        )
    print("Subset validation:       PASS")


if __name__ == "__main__":
    main()
