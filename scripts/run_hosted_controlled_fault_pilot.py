from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from cnsm_agentic.autonomous_research.hosted_controlled_fault_pilot import (
    HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY,
    HostedControlledFaultPilot,
    _repair_prompt,
    _source_prompt,
    controlled_fault_plan_issues,
)
from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    generate_task,
    inject_controlled_fault,
    render_reference_configuration,
)


def _header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _verify_environment(model: str) -> None:
    if load_dotenv is not None:
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not available.")
    from openai import OpenAI

    client = OpenAI()
    model_object = client.models.retrieve(model)
    print(
        "Model availability:      PASS "
        f"({getattr(model_object, 'id', model)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--task-indices", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--show-prompts", action="store_true")
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        indices = [
            int(item.strip())
            for item in args.task_indices.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--task-indices must contain integers.") from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_id = f"controlled-fault-pilot-{timestamp}"
    run_dir = (
        args.run_dir
        or Path("studies/development")
        / f"{timestamp}-controlled-fault-pilot"
    ).resolve()
    plan = {
        "study_id": study_id,
        "adapter_family": HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY,
        "execution_mode": "scientific_pilot",
        "conditions": ["baseline", "guarded"],
        "task_indices": indices,
        "maximum_model_calls": len(indices) * 2,
        "model_provider": "openai_responses",
        "model_name": args.model,
        "maximum_attempts_per_call": 1,
        "reasoning_effort": "minimal",
        "max_output_tokens": args.max_output_tokens,
    }
    issues = controlled_fault_plan_issues(plan)
    if issues:
        raise SystemExit("Plan validation failed:\n- " + "\n- ".join(issues))

    _header("CONTROLLED-FAULT PILOT PREFLIGHT")
    print(f"Execution mode:         {plan['execution_mode']}")
    print(f"Adapter:                {plan['adapter_family']}")
    print(f"Model:                  {plan['model_name']}")
    print("Task indices:           " + ",".join(map(str, indices)))
    print(f"Task count:             {len(indices)}")
    print(f"Terminal episodes:      {len(indices) * 2}")
    print(f"Maximum model calls:    {plan['maximum_model_calls']}")
    print("Source generation/task: 1")
    print("Fault injection/task:   deterministic, one transformation")
    print("Baseline repair calls:  0")
    print("Guarded repair/task:    exactly 1 after valid source")
    print(f"Max output tokens/call: {plan['max_output_tokens']}")
    print("Reasoning effort:       minimal")
    print(f"Fresh run directory:    {run_dir}")
    print("Final-run claim:        NO")

    if args.show_prompts:
        for index in indices:
            task = generate_task(index)
            injected = inject_controlled_fault(
                task,
                render_reference_configuration(task),
            )
            _header(f"TASK {index}: SOURCE GENERATION PROMPT")
            print(_source_prompt(task))
            _header(f"TASK {index}: REPAIR PROMPT TEMPLATE")
            print(
                _repair_prompt(
                    task,
                    injected["injected_configuration"],
                    injected["injected_validation"],
                )
            )

    if not args.execute_paid:
        _header("NO PAID CALLS MADE")
        print("Preflight completed.")
        return

    _verify_environment(args.model)
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "pilot_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = HostedControlledFaultPilot().execute(
        plan=plan,
        execution_dir=run_dir / "execution",
    )

    _header("PAID CONTROLLED-FAULT PILOT COMPLETED")
    print(f"Run directory:          {run_dir}")
    print(f"Study ID:               {summary['study_id']}")
    print(f"Execution status:       {summary['execution_status']}")
    print(f"Model calls used:       {summary['model_calls_used']}")
    print(f"Maximum model calls:    {summary['maximum_model_calls']}")
    print(f"Input tokens:           {summary['input_tokens']}")
    print(f"Output tokens:          {summary['output_tokens']}")
    print(f"Total tokens:           {summary['total_tokens']}")
    print(f"Complete pairs:         {summary['complete_pairs']}")
    print(f"Baseline successes:     {summary['baseline_successes']}")
    print(f"Guarded successes:      {summary['guarded_successes']}")
    print(f"n_10 guarded-only:      {summary['n_10_guarded_only']}")
    print(f"n_01 baseline-only:     {summary['n_01_baseline_only']}")
    print(f"Paired difference:      {summary['paired_difference']}")


if __name__ == "__main__":
    main()
