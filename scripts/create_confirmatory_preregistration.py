from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ORIGINAL_SAMPLE_SHA256 = (
    "5fcdf05f33884a1aa6d9b634219f549c"
    "ec08b88de601392d00eb7b3b0b0c8116"
)

PERMUTED_SAMPLE_SHA256 = (
    "f9fc4f8a47616d253de043cc7cd48df"
    "21c0f050c5c4cabd0a2ca97341780859f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the frozen preregistration for the "
            "6G-Bench confirmatory permutation experiment."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    args = parse_args()

    samples_directory = (
        args.source_run
        / "datasets"
        / "samples"
    )

    original_sample = (
        samples_directory
        / "6g_bench_confirmatory_10_per_task.jsonl"
    )

    permuted_sample = (
        samples_directory
        / (
            "6g_bench_confirmatory_"
            "10_per_task_permuted.jsonl"
        )
    )

    permutation_manifest = (
        samples_directory
        / (
            "6g_bench_confirmatory_"
            "10_per_task_permuted_manifest.json"
        )
    )

    output_directory = (
        args.source_run
        / "experiments"
        / "confirmatory"
    )

    output_path = (
        output_directory
        / "preregistration.json"
    )

    for path in [
        original_sample,
        permuted_sample,
        permutation_manifest,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    if output_path.exists():
        raise FileExistsError(
            f"Preregistration already exists: {output_path}"
        )

    actual_original_hash = sha256_file(
        original_sample
    )

    actual_permuted_hash = sha256_file(
        permuted_sample
    )

    if actual_original_hash != ORIGINAL_SAMPLE_SHA256:
        raise ValueError(
            "Original confirmatory sample hash mismatch."
        )

    if actual_permuted_hash != PERMUTED_SAMPLE_SHA256:
        raise ValueError(
            "Permuted confirmatory sample hash mismatch."
        )

    preregistration = {
        "study_name": (
            "6G-Bench option-permutation "
            "confirmatory experiment"
        ),
        "created_at_utc": utc_now(),
        "status": "frozen_before_model_execution",
        "pilot_status": (
            "The preceding 60-question experiment is "
            "exploratory and is not included in the "
            "confirmatory statistical test."
        ),
        "benchmark": "6G-Bench",
        "model": "gpt-5-nano",
        "sample_design": {
            "record_count": 300,
            "task_count": 30,
            "records_per_task": 10,
            "pilot_question_exclusion_count": 60,
            "overlap_with_pilot_count": 0,
            "original_sample": str(
                original_sample.relative_to(
                    args.source_run
                )
            ),
            "original_sample_sha256": (
                actual_original_hash
            ),
            "permuted_sample": str(
                permuted_sample.relative_to(
                    args.source_run
                )
            ),
            "permuted_sample_sha256": (
                actual_permuted_hash
            ),
            "permutation_manifest": str(
                permutation_manifest.relative_to(
                    args.source_run
                )
            ),
            "permutation_seed": 20260729,
            "identity_permutation_count": 0,
        },
        "execution_conditions": {
            "run_count": 3,
            "total_model_calls": 900,
            "runs": [
                {
                    "name": "original",
                    "run_label": (
                        "confirmatory-original-n300"
                    ),
                    "sample": str(
                        original_sample.relative_to(
                            args.source_run
                        )
                    ),
                },
                {
                    "name": "repeat",
                    "run_label": (
                        "confirmatory-repeat-n300"
                    ),
                    "sample": str(
                        original_sample.relative_to(
                            args.source_run
                        )
                    ),
                },
                {
                    "name": "permuted",
                    "run_label": (
                        "confirmatory-permuted-n300"
                    ),
                    "sample": str(
                        permuted_sample.relative_to(
                            args.source_run
                        )
                    ),
                },
            ],
            "model_settings": {
                "reasoning_effort": "minimal",
                "verbosity": "low",
                "max_tokens": 256,
                "temperature": (
                    "SDK/provider default; must not be "
                    "changed between runs"
                ),
                "prompt": (
                    "Answer the supplied multiple-choice "
                    "question. Return exactly one option "
                    "label: A, B, C, or D. Return no "
                    "explanation, punctuation, or "
                    "additional text."
                ),
            },
            "record_order": (
                "Preserve the JSONL order in every run."
            ),
            "parser_policy": (
                "Use the frozen parser implemented before "
                "confirmatory execution. Do not modify it "
                "between runs."
            ),
        },
        "primary_hypothesis": {
            "null": (
                "Semantic disagreement between the "
                "original and permuted runs equals semantic "
                "disagreement between the original and "
                "unchanged-repeat runs."
            ),
            "alternative": (
                "Option permutation produces greater "
                "semantic disagreement than an unchanged "
                "repeat."
            ),
            "primary_effect": (
                "permutation_disagreement_rate minus "
                "repeat_disagreement_rate"
            ),
            "primary_paired_counts": [
                "permutation_only_disagrees",
                "repeat_only_disagrees",
            ],
            "primary_test": (
                "Exact two-sided McNemar test over the "
                "paired question-level disagreement "
                "indicators."
            ),
            "significance_level": 0.05,
        },
        "primary_endpoints": [
            "repeat_disagreement_count",
            "repeat_disagreement_rate",
            "permutation_disagreement_count",
            "permutation_disagreement_rate",
            "absolute_excess_disagreement",
            "relative_disagreement_ratio",
            "permutation_only_disagrees",
            "repeat_only_disagrees",
            "exact_mcnemar_two_sided_p",
        ],
        "secondary_endpoints": [
            "original_accuracy",
            "repeat_accuracy",
            "permuted_accuracy",
            "coverage for all three runs",
            "semantic agreement rates",
            "label agreement rates",
            "correct-to-wrong transitions",
            "wrong-to-correct transitions",
            "instability by task",
            "instability by original gold option",
            "prediction-label distributions",
            "latency and token usage",
        ],
        "cluster_aware_analysis": {
            "unit": "task",
            "cluster_count": 30,
            "method": (
                "Bootstrap the 30 tasks with replacement, "
                "retaining all 10 questions in each sampled "
                "task."
            ),
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": 20260730,
            "reported_interval": (
                "Percentile 95% confidence interval for "
                "the excess disagreement effect."
            ),
        },
        "missing_data_policy": {
            "model_call_failure": (
                "Retain and report the failed record. "
                "Do not silently rerun individual items."
            ),
            "unparsed_output": (
                "Count as uncovered and exclude from "
                "accuracy-on-parsed, while reporting "
                "coverage separately."
            ),
            "whole_run_interruption": (
                "Resume only from records not already "
                "written to predictions.jsonl. Do not "
                "repeat completed questions."
            ),
        },
        "analysis_freeze": {
            "no_prompt_changes_after_start": True,
            "no_parser_changes_after_start": True,
            "no_sample_changes_after_start": True,
            "no_permutation_changes_after_start": True,
            "no_selective_item_reruns": True,
            "pilot_results_not_pooled": True,
        },
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            preregistration,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    preregistration_hash = sha256_file(
        output_path
    )

    print(
        json.dumps(
            {
                "status": preregistration["status"],
                "record_count": 300,
                "total_model_calls": 900,
                "original_sample_sha256": (
                    actual_original_hash
                ),
                "permuted_sample_sha256": (
                    actual_permuted_hash
                ),
                "preregistration_sha256": (
                    preregistration_hash
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved preregistration: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
