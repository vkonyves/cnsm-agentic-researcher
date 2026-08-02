from __future__ import annotations

import argparse
from pathlib import Path

from cnsm_agentic.autonomous_research.controlled_fault_analysis import analyze_controlled_fault_execution


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a hosted controlled-fault NetOps execution.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    result = analyze_controlled_fault_execution(
        run_dir / "execution",
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        persist=True,
    )
    bootstrap = result["bootstrap"]
    print()
    print("CONTROLLED-FAULT PAIRED ANALYSIS")
    print("================================")
    print(f"Run directory:          {run_dir}")
    print(f"Study ID:               {result['study_id']}")
    print(f"Complete pairs:         {result['complete_pair_count']}")
    print(f"Incomplete pairs:       {result['incomplete_pair_count']}")
    print(f"Baseline successes:     {result['baseline_successes']}")
    print(f"Guarded successes:      {result['guarded_successes']}")
    print(f"n_10 guarded-only:      {result['n_10_guarded_only']}")
    print(f"n_01 baseline-only:     {result['n_01_baseline_only']}")
    print(f"Paired difference:      {result['paired_difference']:.6f}")
    print(f"Exact McNemar p-value:  {result['exact_mcnemar_p_value']:.6f}")
    print(
        "Bootstrap CI:           "
        f"[{bootstrap['paired_difference_ci_low']:.6f}, "
        f"{bootstrap['paired_difference_ci_high']:.6f}]"
    )
    print("Analysis validation:    PASS")


if __name__ == "__main__":
    main()
