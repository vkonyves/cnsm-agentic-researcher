from __future__ import annotations

import argparse
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_launch_audit import (
    audit_controlled_fault_launch,
    write_launch_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a frozen controlled-fault launch without making "
            "provider calls."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2000,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = audit_controlled_fault_launch(
        plan_path=args.plan,
        repository_root=args.repository_root,
        output_path=args.output,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
    )
    write_launch_audit(
        report,
        args.output,
        overwrite=args.overwrite,
    )

    print()
    print("CONTROLLED-FAULT LAUNCH AUDIT")
    print("==============================")
    print(f"Status:                  {report['status']}")
    print(f"Plan:                    {report['plan_path']}")
    print(f"Plan SHA-256:            {report['plan_sha256']}")
    print(f"Pairs audited:           {report['pair_count']}")
    print(
        "Maximum model calls:    "
        f"{report['maximum_model_calls']}"
    )
    print(
        "Computed call ceiling:  "
        f"{report['computed_maximum_model_calls']}"
    )
    print(
        "Source prompt hashes:   "
        f"{report['source_prompt_count']}"
    )
    print(
        "Repair prompt hashes:   "
        f"{report['repair_prompt_count']}"
    )
    print(
        "Code fingerprint:       "
        f"{report['runner']['code_fingerprint_sha256']}"
    )
    print(
        "Aggregate prompt hash:  "
        f"{report['aggregate_prompt_sha256']}"
    )
    print(
        "Aggregate fault hash:   "
        f"{report['aggregate_shared_candidate_sha256']}"
    )
    print(f"Audit report SHA-256:    {report['audit_report_sha256']}")
    print(f"Provider initialized:    {report['provider_initialized']}")
    print(f"Provider calls made:     {report['provider_calls_made']}")
    print(f"Output:                  {report['output_path']}")

    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("Launch validation:       PASS")
    print()
    print("NO PAID CALLS MADE")


if __name__ == "__main__":
    main()
