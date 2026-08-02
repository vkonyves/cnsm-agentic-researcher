from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    load_experiment_plan,
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
    args = parser.parse_args()

    plan = load_experiment_plan(args.plan.resolve())
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
    print(f"Plan:                   {args.plan.resolve()}")
    print(f"Plan SHA-256:           {plan['plan_sha256']}")
    print(f"Pairs:                  {plan['pair_count']}")
    print(f"Maximum model calls:    {plan['maximum_model_calls']}")
    print(f"Model:                  {args.model}")
    print(f"Max output tokens:      {args.max_output_tokens}")
    print(f"Run directory:          {run_dir}")
    print(f"Resume requested:       {args.resume}")
    print("Final-run claim:        NO")

    if not args.execute_paid:
        print()
        print("NO PAID CALLS MADE")
        print("==================")
        print("Preflight completed.")
        return

    _verify_environment(args.model)
    summary = HostedControlledFaultPlanRunner().execute(
        plan_path=args.plan,
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
