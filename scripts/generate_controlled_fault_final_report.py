from __future__ import annotations
import argparse
from pathlib import Path
from cnsm_agentic.autonomous_research.controlled_fault_final_report import build_final_experiment_report, write_final_experiment_report

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a read-only final report from a frozen controlled-fault run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launch-lock", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_final_experiment_report(run_dir=args.run_dir, launch_lock_path=args.launch_lock, audit_path=args.audit, bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed)
    write_final_experiment_report(report, json_path=args.json_output, markdown_path=args.markdown_output, overwrite=args.overwrite)
    analysis = report["paired_analysis"]
    execution = report["execution"]
    print("\nCONTROLLED-FAULT FINAL EXPERIMENT REPORT")
    print("========================================")
    print(f"Status:                  {report['status']}")
    print(f"Complete pairs:          {execution['complete_pair_count']}")
    print(f"Incomplete pairs:        {execution['incomplete_pair_count']}")
    print(f"Baseline successes:      {analysis['baseline_successes']}")
    print(f"Guarded successes:       {analysis['guarded_successes']}")
    print(f"Paired difference:       {analysis['paired_difference']:.6f}")
    print(f"Exact McNemar p-value:  {analysis['exact_mcnemar_p_value']:.12g}")
    print(f"Bootstrap CI:           [{analysis['bootstrap_ci_95'][0]:.6f}, {analysis['bootstrap_ci_95'][1]:.6f}]")
    print(f"Failed repairs:          {len(report['failed_repairs'])}")
    print(f"JSON output:             {args.json_output.resolve()}")
    print(f"Markdown output:         {args.markdown_output.resolve()}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("Report validation:       PASS")

if __name__ == "__main__":
    main()
