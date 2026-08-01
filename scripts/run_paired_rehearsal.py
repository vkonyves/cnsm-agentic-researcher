from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cnsm_agentic.autonomous_research.analysis_executors import (
    PairedBinaryAnalysisExecutor,
    validate_analysis_results,
)
from cnsm_agentic.autonomous_research.execution_adapters import (
    SyntheticPairedLLMBenchmarkAdapter,
    validate_execution_manifest,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_inventory(run_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and validate the deterministic paired LLM benchmark "
            "development rehearsal."
        )
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=6,
        help="Number of paired tasks. Default: 6.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Pair-level bootstrap resamples. Default: 10000.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=7,
        help="Deterministic bootstrap seed. Default: 7.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory. By default a fresh timestamped directory "
            "is created under studies/development."
        ),
    )
    args = parser.parse_args()

    if args.task_count <= 0:
        raise SystemExit("--task-count must be positive.")
    if args.bootstrap_resamples <= 0:
        raise SystemExit("--bootstrap-resamples must be positive.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else Path("studies") / "development" / f"{timestamp}-paired-rehearsal"
    )
    run_dir = run_dir.resolve()

    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory is not empty: {run_dir}")

    execution_dir = run_dir / "execution"
    study_id = f"paired-rehearsal-{timestamp}"

    plan: dict[str, Any] = {
        "study_id": study_id,
        "adapter_family": "synthetic_paired_llm_benchmark_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "task_count": args.task_count,
        "estimated_model_calls": args.task_count * 2,
        "task_families": ["configuration_error_detection_v1"],
        "transformations": {
            "baseline": "baseline_prompt_v1",
            "guarded": "guarded_prompt_v1",
        },
        "model_provider": "deterministic_local",
        "model_name": "paired-smoke-model",
        "model_version": "1.0",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "development_rehearsal",
        "rehearsal_started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    preregistration: dict[str, Any] = {
        "study_id": study_id,
        "execution_mode": "development_rehearsal",
        "primary_estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "failed_call_treatment": "complete_pair_primary",
        "note": (
            "Deterministic development rehearsal only; not scientific evidence."
        ),
    }

    adapter = SyntheticPairedLLMBenchmarkAdapter()
    if not adapter.supports(plan):
        raise SystemExit("Concrete execution adapter rejected the rehearsal plan.")

    execution_manifest = adapter.execute(
        plan=plan,
        preregistration=preregistration,
        output_dir=execution_dir,
    )
    execution_issues = validate_execution_manifest(
        execution_manifest,
        plan=plan,
        output_dir=execution_dir,
        maximum_model_calls=args.task_count * 2,
    )
    if execution_issues:
        raise SystemExit(
            "Execution validation failed:\n- "
            + "\n- ".join(execution_issues)
        )

    analysis_plan: dict[str, Any] = {
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

    executor = PairedBinaryAnalysisExecutor()
    if not executor.supports(
        analysis_plan=analysis_plan,
        execution_manifest=execution_manifest,
    ):
        raise SystemExit("Concrete analysis executor rejected the execution.")

    analysis_results = executor.execute(
        analysis_plan=analysis_plan,
        preregistration=preregistration,
        execution_manifest=execution_manifest,
        run_dir=run_dir,
    )
    analysis_issues = validate_analysis_results(
        analysis_results,
        run_dir=run_dir,
        execution_manifest=execution_manifest,
    )
    if analysis_issues:
        raise SystemExit(
            "Analysis validation failed:\n- "
            + "\n- ".join(analysis_issues)
        )

    persisted_results_path = run_dir / analysis_results["results_path"]
    persisted_results = _load_json(persisted_results_path)
    if persisted_results != analysis_results:
        raise SystemExit(
            "Persisted analysis results differ from returned analysis results."
        )

    reloaded_issues = validate_analysis_results(
        persisted_results,
        run_dir=run_dir,
        execution_manifest=execution_manifest,
    )
    if reloaded_issues:
        raise SystemExit(
            "Reloaded analysis validation failed:\n- "
            + "\n- ".join(reloaded_issues)
        )

    primary = analysis_results["confirmatory_results"][0]

    print()
    print("DEVELOPMENT REHEARSAL COMPLETED")
    print("=" * 34)
    print(f"Run directory:          {run_dir}")
    print(f"Study ID:               {study_id}")
    print(f"Execution status:       {execution_manifest['status']}")
    print(
        "Episodes:               "
        f"{execution_manifest['completed_episode_count']} completed, "
        f"{execution_manifest['failed_episode_count']} failed, "
        f"{execution_manifest['planned_episode_count']} planned"
    )
    print(f"Model calls used:       {execution_manifest['model_calls_used']}")
    print(f"Complete pairs:         {primary['complete_pair_count']}")
    print(f"Baseline successes:     {primary['baseline_success_count']}")
    print(f"Guarded successes:      {primary['guarded_success_count']}")
    print(f"n_10 guarded-only:      {primary['n_10']}")
    print(f"n_01 baseline-only:     {primary['n_01']}")
    print(f"Paired difference:      {primary['estimate']:.6f}")
    print(f"Exact McNemar p-value:  {primary['p_value']:.6f}")
    print(
        "95% bootstrap CI:      "
        f"[{primary['confidence_interval_lower']:.6f}, "
        f"{primary['confidence_interval_upper']:.6f}]"
    )
    print()
    print("VALIDATION")
    print("Execution manifest:     PASS")
    print("Analysis result:        PASS")
    print("Persisted reload:       PASS")
    print()
    print("ARTIFACTS")
    for relative_path in _relative_inventory(run_dir):
        print(f"- {relative_path}")
    print()
    print(
        "WARNING: This run used a deterministic local stand-in and is "
        "not scientific evidence."
    )


if __name__ == "__main__":
    main()


