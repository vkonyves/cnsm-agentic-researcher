from __future__ import annotations

import argparse
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_launch_lock import (
    create_launch_lock,
    validate_launch_lock,
    write_launch_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate a launch lock for the frozen "
            "controlled-fault execution."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-head-tag")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    if args.validate_existing:
        import json

        lock = json.loads(args.output.read_text(encoding="utf-8"))
        issues = validate_launch_lock(
            lock,
            repository_root=args.repository_root,
            plan_path=args.plan,
            audit_path=args.audit,
            intended_run_dir=args.run_dir,
            model_name=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        print()
        print("CONTROLLED-FAULT LAUNCH LOCK VALIDATION")
        print("========================================")
        print(f"Lock:                    {args.output.resolve()}")
        print(f"Validation issues:       {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        if issues:
            raise SystemExit(1)
        print("Launch-lock validation:  PASS")
        return

    lock = create_launch_lock(
        plan_path=args.plan,
        audit_path=args.audit,
        repository_root=args.repository_root,
        intended_run_dir=args.run_dir,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        require_clean_worktree=True,
        require_head_tag=args.require_head_tag,
    )
    write_launch_lock(lock, args.output, overwrite=args.overwrite)

    print()
    print("CONTROLLED-FAULT LAUNCH LOCK")
    print("=============================")
    print(f"Status:                  {lock['status']}")
    print(f"Plan SHA-256:            {lock['plan_sha256']}")
    print(f"Audit SHA-256:           {lock['audit_report_sha256']}")
    print(
        "Code fingerprint:       "
        f"{lock['runner']['code_fingerprint_sha256']}"
    )
    print(f"Git commit:              {lock['git']['commit_sha']}")
    print(f"Git branch:              {lock['git']['branch']}")
    print(
        "Git tags at HEAD:        "
        f"{','.join(lock['git']['tags_at_head']) or '(none)'}"
    )
    print(
        "Working tree clean:      "
        f"{lock['git']['working_tree_clean']}"
    )
    print(f"Intended run directory:  {lock['intended_run_dir']}")
    print(f"Maximum model calls:     {lock['maximum_model_calls']}")
    print(f"Launch-lock SHA-256:     {lock['launch_lock_sha256']}")
    print(f"Output:                  {args.output.resolve()}")

    if lock["issues"]:
        print("Issues:")
        for issue in lock["issues"]:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("Launch lock:             PASS")


if __name__ == "__main__":
    main()
