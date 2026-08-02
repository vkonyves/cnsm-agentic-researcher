from __future__ import annotations

import argparse
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    DEFAULT_PAIR_COUNT,
    DEFAULT_SEED,
    generate_experiment_plan,
    write_experiment_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and freeze a controlled-fault experiment plan."
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=DEFAULT_PAIR_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/plans/controlled-fault-plan-v1.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    plan = generate_experiment_plan(
        pair_count=args.pair_count,
        seed=args.seed,
    )
    output = write_experiment_plan(
        plan,
        args.output,
        overwrite=args.overwrite,
    )

    print()
    print("CONTROLLED-FAULT EXPERIMENT PLAN")
    print("================================")
    print(f"Output:                  {output}")
    print(f"Plan SHA-256:            {plan['plan_sha256']}")
    print(f"Pairs:                   {plan['pair_count']}")
    print(f"Maximum model calls:     {plan['maximum_model_calls']}")
    print(
        "Faults per class:        "
        f"{plan['fault_class_target_count']}"
    )
    print(
        "Tasks per pattern:       "
        f"{plan['workflow_pattern_target_count']}"
    )
    print("Fault class counts:")
    for name, count in plan["fault_class_counts"].items():
        print(f"  {name}: {count}")
    print("Workflow pattern counts:")
    for name, count in plan["workflow_pattern_counts"].items():
        print(f"  {name}: {count}")
    print("Plan validation:         PASS")


if __name__ == "__main__":
    main()
