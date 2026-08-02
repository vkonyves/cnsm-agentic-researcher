from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    load_experiment_plan,
)
from cnsm_agentic.autonomous_research.controlled_fault_launch_lock import (
    validate_launch_lock,
)
from cnsm_agentic.autonomous_research.hosted_controlled_fault_plan_runner import (
    HostedControlledFaultPlanRunner,
    PLAN_RUNNER_ID,
    PLAN_RUNNER_VERSION,
)


def _verify_environment(model: str) -> None:
    if load_dotenv is not None:
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not available.")
    from openai import OpenAI

    model_object = OpenAI().models.retrieve(model)
    print(
        "Model availability:      PASS "
        f"({getattr(model_object, 'id', model)})"
    )


def _read_launch_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Launch lock does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Launch lock is not valid JSON: {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise SystemExit("Launch lock must contain a JSON object.")
    return value


def _validate_paid_launch(
    *,
    launch_lock_path: Path | None,
    repository_root: Path,
    plan_path: Path,
    run_dir: Path,
    model_name: str,
    max_output_tokens: int,
    final_run: bool,
) -> dict[str, Any]:
    if launch_lock_path is None:
        mode = "final execution" if final_run else "paid execution"
        raise SystemExit(
            f"A validated --launch-lock is required for {mode}."
        )

    lock_path = launch_lock_path.resolve()
    lock = _read_launch_lock(lock_path)

    if lock.get("status") != "LOCKED":
        raise SystemExit(
            "Launch lock status is not LOCKED: "
            f"{lock.get('status')!r}."
        )

    audit_path_raw = lock.get("audit_path")
    if not isinstance(audit_path_raw, str) or not audit_path_raw:
        raise SystemExit("Launch lock does not contain a valid audit_path.")

    audit_path = Path(audit_path_raw)
    issues = validate_launch_lock(
        lock,
        repository_root=repository_root.resolve(),
        plan_path=plan_path.resolve(),
        audit_path=audit_path,
        intended_run_dir=run_dir.resolve(),
        model_name=model_name,
        max_output_tokens=max_output_tokens,
    )
    if issues:
        raise SystemExit(
            "Launch-lock validation failed:\n- "
            + "\n- ".join(issues)
        )

    print(f"Launch lock:            PASS ({lock_path})")
    print(f"Launch-lock SHA-256:    {lock.get('launch_lock_sha256')}")
    print(f"Locked Git commit:      {lock['git']['commit_sha']}")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute or resume a frozen controlled-fault plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--launch-lock", type=Path)
    parser.add_argument("--final-run", action="store_true")
    args = parser.parse_args()

    if args.final_run and not args.execute_paid:
        raise SystemExit("--final-run requires --execute-paid.")

    plan_path = args.plan.resolve()
    plan = load_experiment_plan(plan_path)
    repository_root = Path(".").resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (
            Path("studies/development")
            / f"{timestamp}-controlled-fault-plan-run"
        ).resolve()
    )

    print()
    print("FROZEN CONTROLLED-FAULT PLAN PREFLIGHT")
    print("=======================================")
    print(f"Runner:                 {PLAN_RUNNER_ID}")
    print(f"Runner version:         {PLAN_RUNNER_VERSION}")
    print(f"Plan:                   {plan_path}")
    print(f"Plan SHA-256:           {plan['plan_sha256']}")
    print(f"Pairs:                  {plan['pair_count']}")
    print(f"Maximum model calls:    {plan['maximum_model_calls']}")
    print(f"Model:                  {args.model}")
    print(f"Max output tokens:      {args.max_output_tokens}")
    print(f"Run directory:          {run_dir}")
    print(f"Resume requested:       {args.resume}")
    print(
        "Final-run claim:        "
        f"{'YES' if args.final_run else 'NO'}"
    )

    if not args.execute_paid:
        print()
        print("NO PAID CALLS MADE")
        print("==================")
        print("Preflight completed.")
        return

    lock = _validate_paid_launch(
        launch_lock_path=args.launch_lock,
        repository_root=repository_root,
        plan_path=plan_path,
        run_dir=run_dir,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        final_run=args.final_run,
    )

    if args.final_run:
        if args.resume:
            raise SystemExit(
                "A new --final-run cannot start with --resume. "
                "Resume an interrupted final run without --final-run."
            )
        if run_dir.exists():
            raise SystemExit(
                "Final-run directory already exists; refusing to overwrite "
                f"or ambiguously resume it: {run_dir}"
            )
        print("Final launch gate:      PASS")
        print(
            "Locked audit SHA-256:   "
            f"{lock['audit_report_sha256']}"
        )
        print(
            "Locked plan SHA-256:    "
            f"{lock['plan_sha256']}"
        )

    _verify_environment(args.model)
    summary = HostedControlledFaultPlanRunner().execute(
        plan_path=plan_path,
        run_dir=run_dir,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
    )

    print()
    print("PLAN-DRIVEN CONTROLLED-FAULT RUN")
    print("================================")
    print(f"Execution status:       {summary['execution_status']}")
    print(f"Terminal pairs:         {summary['terminal_pair_count']}")
    print(f"Completed pairs:        {summary['completed_pair_count']}")
    print(
        "Scientific pairs:       "
        f"{summary['complete_scientific_pair_count']}"
    )
    print(f"Model calls used:       {summary['model_calls_used']}")
    print(f"Maximum model calls:    {summary['maximum_model_calls']}")
    print(f"Baseline successes:     {summary['baseline_successes']}")
    print(f"Guarded successes:      {summary['guarded_successes']}")
    print(f"n_10 guarded-only:      {summary['n_10_guarded_only']}")
    print(f"n_01 baseline-only:     {summary['n_01_baseline_only']}")
    print(f"Paired difference:      {summary['paired_difference']}")


if __name__ == "__main__":
    main()
