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

from cnsm_agentic.autonomous_research.analysis_executors import (
    PairedBinaryAnalysisExecutor,
    validate_analysis_results,
)
from cnsm_agentic.autonomous_research.execution_adapters import (
    validate_execution_manifest,
)
from cnsm_agentic.autonomous_research.hosted_netops_adapter import (
    HostedNetOpsGVRAdapter,
    _repair_prompt,
    _task_prompt,
    hosted_netops_plan_issues,
)
from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    generate_task,
)


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TASK_COUNT = 2
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 7


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _build_plan(
    *,
    study_id: str,
    model: str,
    task_indices: list[int],
    max_output_tokens: int,
) -> dict[str, Any]:
    task_count = len(task_indices)
    maximum_calls = task_count * 2
    return {
        "study_id": study_id,
        "adapter_family": "hosted_netops_gvr_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "task_count": task_count,
        "task_indices": task_indices,
        "estimated_model_calls": maximum_calls,
        "maximum_model_calls": maximum_calls,
        "task_families": ["intent_configuration_repair_v1"],
        "transformations": {
            "baseline": "direct_configuration_generation_v1",
            "guarded": "generate_validate_repair_v1",
        },
        "model_provider": "openai_responses",
        "model_name": model,
        "model_version": model,
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "scientific_pilot",
        "maximum_attempts_per_call": 1,
        "max_output_tokens": max_output_tokens,
        "temperature": None,
        "reasoning_effort": "minimal",
        "retry_backoff_seconds": 0.0,
    }


def _build_preregistration(
    *,
    study_id: str,
    model: str,
    task_count: int,
    maximum_model_calls: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": study_id,
        "execution_mode": "scientific_pilot",
        "research_question": (
            "Does deterministic validation with one bounded LLM repair "
            "increase the rate of valid intent-satisfying NetOps "
            "configurations compared with direct LLM generation?"
        ),
        "primary_estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "primary_test": "exact_mcnemar_two_sided",
        "failed_call_treatment": "complete_pair_primary",
        "conditions": ["baseline", "guarded"],
        "task_count": task_count,
        "model": model,
        "maximum_model_calls": maximum_model_calls,
        "maximum_attempts_per_call": 1,
        "maximum_guarded_repair_calls_per_task": 1,
        "shared_initial_candidate_per_pair": True,
        "scorer": "deterministic_netops_validator_v1",
        "human_scientific_intervention_after_launch": False,
        "note": (
            "Small genuine hosted-model scientific pilot. This is not the "
            "final conference-compliant autonomous run."
        ),
    }


def _print_prompt_preview(task_indices: list[int]) -> None:
    for index in task_indices:
        task = generate_task(index)
        _print_header(f"TASK {index}: INITIAL GENERATION PROMPT")
        print(_task_prompt(task))
        _print_header(f"TASK {index}: POSSIBLE REPAIR PROMPT TEMPLATE")
        example_candidate = "interface ethX mtu 1400"
        example_validation = {
            "violations": [
                {
                    "code": "INTENT_CONSTRAINT_VIOLATION",
                    "message": (
                        "Example placeholder shown during preflight only."
                    ),
                }
            ]
        }
        print(
            _repair_prompt(
                task,
                example_candidate,
                example_validation,
            )
        )


def _verify_environment(model: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not available. Add it to the environment "
            "or repository .env file before paid execution."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai package is not installed in this environment."
        ) from exc

    client = OpenAI()
    try:
        model_object = client.models.retrieve(model)
    except Exception as exc:
        raise SystemExit(
            f"Could not verify API access to model {model!r}: {exc}"
        ) from exc

    print(
        "Model availability:      PASS "
        f"({getattr(model_object, 'id', model)})"
    )


def _print_preflight(
    *,
    plan: dict[str, Any],
    run_dir: Path,
) -> None:
    _print_header("SCIENTIFIC PILOT PREFLIGHT")
    print(f"Execution mode:         {plan['execution_mode']}")
    print(f"Adapter:                {plan['adapter_family']}")
    print(f"Model:                  {plan['model_name']}")
    print(f"Task count:             {plan['task_count']}")
    print(
        "Task indices:           "
        + ",".join(str(item) for item in plan["task_indices"])
    )
    print(f"Terminal episodes:      {plan['task_count'] * 2}")
    print(f"Maximum model calls:    {plan['maximum_model_calls']}")
    print("Attempts per call:      1")
    print("Shared generation/task: 1")
    print("Guarded repair/task:    at most 1")
    print(f"Max output tokens/call: {plan['max_output_tokens']}")
    print("Temperature:            provider default")
    print("Reasoning effort:       minimal")
    print("API response storage:   store=False")
    print(f"Fresh run directory:    {run_dir}")
    print(f"Call cache directory:   {run_dir / 'execution/call_cache'}")
    print("Final-run claim:        NO")
    print("Human paper lock:       NOT ACTIVATED (pilot only)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute a bounded two-task hosted NetOps "
            "generate-validate-repair scientific pilot."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI API model identifier. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=DEFAULT_TASK_COUNT,
        help="Number of paired tasks. Default: 2.",
    )
    parser.add_argument(
        "--task-indices",
        default=None,
        help=(
            "Comma-separated deterministic task indices, for example 7,8. "
            "When omitted, uses 1..task-count."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for each API call. Default: 240.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print exact initial prompts and a repair-template preview.",
    )
    parser.add_argument(
        "--check-api-access",
        action="store_true",
        help=(
            "Check OPENAI_API_KEY and retrieve the selected model. "
            "This makes no generation call."
        ),
    )
    parser.add_argument(
        "--execute-paid",
        action="store_true",
        help=(
            "Explicitly permit real hosted generation calls. Without this "
            "flag the script performs preflight only."
        ),
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    if args.task_indices:
        try:
            task_indices = [
                int(item.strip())
                for item in args.task_indices.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise SystemExit(
                "--task-indices must be comma-separated positive integers."
            ) from exc
        if (
            not task_indices
            or any(item <= 0 for item in task_indices)
            or len(set(task_indices)) != len(task_indices)
        ):
            raise SystemExit(
                "--task-indices must contain unique positive integers."
            )
        if args.task_count != DEFAULT_TASK_COUNT and (
            args.task_count != len(task_indices)
        ):
            raise SystemExit(
                "--task-count must match the number of --task-indices."
            )
    else:
        if args.task_count <= 0:
            raise SystemExit("--task-count must be positive.")
        task_indices = list(range(1, args.task_count + 1))

    if len(task_indices) > 2:
        raise SystemExit(
            "The first paid pilot is restricted to one or two tasks."
        )
    if args.max_output_tokens <= 0 or args.max_output_tokens > 1000:
        raise SystemExit(
            "--max-output-tokens must be from 1 to 1000 for this pilot."
        )
    if args.bootstrap_resamples <= 0:
        raise SystemExit("--bootstrap-resamples must be positive.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_id = f"hosted-netops-pilot-{timestamp}"
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else Path("studies")
        / "development"
        / f"{timestamp}-hosted-netops-pilot"
    ).resolve()

    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory is not empty: {run_dir}")

    plan = _build_plan(
        study_id=study_id,
        model=args.model,
        task_indices=task_indices,
        max_output_tokens=args.max_output_tokens,
    )
    preregistration = _build_preregistration(
        study_id=study_id,
        model=args.model,
        task_count=len(task_indices),
        maximum_model_calls=plan["maximum_model_calls"],
    )

    issues = hosted_netops_plan_issues(plan)
    if issues:
        raise SystemExit(
            "Pilot plan validation failed:\n- "
            + "\n- ".join(issues)
        )

    _print_preflight(plan=plan, run_dir=run_dir)

    if args.show_prompts:
        _print_prompt_preview(task_indices)

    if args.check_api_access or args.execute_paid:
        _verify_environment(args.model)

    if not args.execute_paid:
        _print_header("NO PAID CALLS MADE")
        print(
            "Preflight completed. Add --execute-paid only after reviewing "
            "the model, prompts, run directory, and four-call ceiling."
        )
        return

    execution_dir = run_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=False)

    preregistration_path = run_dir / "pilot_preregistration.json"
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = run_dir / "pilot_plan.json"
    plan_path.write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = HostedNetOpsGVRAdapter()
    manifest = adapter.execute(
        plan=plan,
        preregistration=preregistration,
        output_dir=execution_dir,
    )
    execution_issues = validate_execution_manifest(
        manifest,
        plan=plan,
        output_dir=execution_dir,
        maximum_model_calls=plan["maximum_model_calls"],
    )
    if execution_issues:
        raise SystemExit(
            "Execution validation failed:\n- "
            + "\n- ".join(execution_issues)
        )

    analysis_plan = {
        "analysis_executor": "paired_binary_analysis_v1",
        "study_id": study_id,
        "estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "failed_call_treatment": "complete_pair_primary",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "confidence_level": 0.95,
    }
    results = PairedBinaryAnalysisExecutor().execute(
        analysis_plan=analysis_plan,
        preregistration=preregistration,
        execution_manifest=manifest,
        run_dir=run_dir,
    )
    analysis_issues = validate_analysis_results(
        results,
        run_dir=run_dir,
        execution_manifest=manifest,
    )
    if analysis_issues:
        raise SystemExit(
            "Analysis validation failed:\n- "
            + "\n- ".join(analysis_issues)
        )

    persisted = _load_json(run_dir / results["results_path"])
    if persisted != results:
        raise SystemExit(
            "Persisted analysis results differ from returned results."
        )

    primary = results["confirmatory_results"][0]
    provider_events = [
        json.loads(line)
        for line in (
            execution_dir / "execution_log.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    total_input_tokens = sum(
        int(event.get("input_tokens") or 0)
        for event in provider_events
    )
    total_output_tokens = sum(
        int(event.get("output_tokens") or 0)
        for event in provider_events
    )
    total_tokens = sum(
        int(event.get("total_tokens") or 0)
        for event in provider_events
    )

    _print_header("PAID SCIENTIFIC PILOT COMPLETED")
    print(f"Run directory:          {run_dir}")
    print(f"Study ID:               {study_id}")
    print(f"Execution status:       {manifest['status']}")
    print(f"Model calls used:       {manifest['model_calls_used']}")
    print(f"Maximum model calls:    {manifest['maximum_model_calls']}")
    print(f"Input tokens:           {total_input_tokens}")
    print(f"Output tokens:          {total_output_tokens}")
    print(f"Total tokens:           {total_tokens}")
    print(f"Complete pairs:         {primary['complete_pair_count']}")
    print(f"Baseline successes:     {primary['baseline_success_count']}")
    print(f"Guarded successes:      {primary['guarded_success_count']}")
    print(f"n_10 guarded-only:      {primary['n_10']}")
    print(f"n_01 baseline-only:     {primary['n_01']}")
    print(f"Paired difference:      {primary['estimate']:.6f}")
    print(f"Exact McNemar p-value:  {primary['p_value']:.6f}")
    print()
    print("Execution validation:   PASS")
    print("Analysis validation:    PASS")
    print("Persisted reload:       PASS")
    print()
    print(
        "This was a small scientific pilot, not the final autonomous "
        "conference run."
    )


if __name__ == "__main__":
    main()
