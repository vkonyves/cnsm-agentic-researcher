from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic disagreement under an unchanged repeat "
            "against disagreement under option permutation."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="gpt-5-nano",
    )

    return parser.parse_args()


def exact_mcnemar_p_value(
    repeat_only_disagrees: int,
    permutation_only_disagrees: int,
) -> float | None:
    discordant = (
        repeat_only_disagrees
        + permutation_only_disagrees
    )

    if discordant == 0:
        return None

    smaller = min(
        repeat_only_disagrees,
        permutation_only_disagrees,
    )

    cumulative = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2**discordant)

    return min(
        1.0,
        2.0 * cumulative,
    )


def main() -> int:
    args = parse_args()

    model_slug = args.model.replace("/", "-")

    repeat_path = (
        args.source_run
        / "experiments"
        / "repeatability_comparison"
        / f"{model_slug}-n60-vs-repeat2-n60.json"
    )

    permutation_path = (
        args.source_run
        / "experiments"
        / "permutation_comparison"
        / f"{model_slug}-comparison.json"
    )

    for path in [repeat_path, permutation_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required comparison not found: {path}"
            )

    repeat_report = json.loads(
        repeat_path.read_text(encoding="utf-8")
    )

    permutation_report = json.loads(
        permutation_path.read_text(encoding="utf-8")
    )

    repeat_pairs = {
        str(item["question_id"]): item
        for item in repeat_report["pairs"]
    }

    permutation_pairs = {
        str(item["original_question_id"]): item
        for item in permutation_report["pairs"]
    }

    if set(repeat_pairs) != set(permutation_pairs):
        raise ValueError(
            "Repeat and permutation comparisons contain "
            "different original question IDs."
        )

    outcome_counts: Counter[str] = Counter()
    task_totals: Counter[str] = Counter()
    task_repeat_disagreements: Counter[str] = Counter()
    task_permutation_disagreements: Counter[str] = Counter()

    pair_results: list[dict[str, object]] = []

    for question_id in sorted(repeat_pairs):
        repeat_pair = repeat_pairs[question_id]
        permutation_pair = permutation_pairs[question_id]

        repeat_disagrees = not bool(
            repeat_pair["semantic_agreement"]
        )

        permutation_disagrees = not bool(
            permutation_pair["semantic_consistency"]
        )

        if not repeat_disagrees and not permutation_disagrees:
            outcome = "both_agree"

        elif repeat_disagrees and not permutation_disagrees:
            outcome = "repeat_only_disagrees"

        elif not repeat_disagrees and permutation_disagrees:
            outcome = "permutation_only_disagrees"

        else:
            outcome = "both_disagree"

        outcome_counts[outcome] += 1

        task_id = str(repeat_pair["task_id"])
        task_totals[task_id] += 1

        if repeat_disagrees:
            task_repeat_disagreements[task_id] += 1

        if permutation_disagrees:
            task_permutation_disagreements[task_id] += 1

        pair_results.append(
            {
                "question_id": question_id,
                "task_id": task_id,
                "task_name": repeat_pair["task_name"],
                "repeat_disagrees": repeat_disagrees,
                "permutation_disagrees": permutation_disagrees,
                "comparison_outcome": outcome,
                "original_prediction": (
                    repeat_pair["first_predicted_option"]
                ),
                "repeat_prediction": (
                    repeat_pair["second_predicted_option"]
                ),
                "permuted_prediction": (
                    permutation_pair[
                        "permuted_predicted_label"
                    ]
                ),
                "original_correct": (
                    repeat_pair["first_correct"]
                ),
                "repeat_correct": (
                    repeat_pair["second_correct"]
                ),
                "permuted_correct": (
                    permutation_pair["permuted_correct"]
                ),
            }
        )

    pair_count = len(pair_results)

    repeat_disagreement_count = sum(
        int(item["repeat_disagrees"])
        for item in pair_results
    )

    permutation_disagreement_count = sum(
        int(item["permutation_disagrees"])
        for item in pair_results
    )

    repeat_only = outcome_counts[
        "repeat_only_disagrees"
    ]

    permutation_only = outcome_counts[
        "permutation_only_disagrees"
    ]

    report = {
        "benchmark": "6G-Bench",
        "model": args.model,
        "pair_count": pair_count,
        "repeat_disagreement_count": (
            repeat_disagreement_count
        ),
        "repeat_disagreement_rate": (
            repeat_disagreement_count / pair_count
        ),
        "permutation_disagreement_count": (
            permutation_disagreement_count
        ),
        "permutation_disagreement_rate": (
            permutation_disagreement_count / pair_count
        ),
        "absolute_excess_disagreement": (
            permutation_disagreement_count
            / pair_count
            - repeat_disagreement_count
            / pair_count
        ),
        "relative_disagreement_ratio": (
            permutation_disagreement_count
            / repeat_disagreement_count
            if repeat_disagreement_count
            else None
        ),
        "paired_outcome_counts": dict(
            sorted(outcome_counts.items())
        ),
        "mcnemar_exact_two_sided_p": (
            exact_mcnemar_p_value(
                repeat_only_disagrees=repeat_only,
                permutation_only_disagrees=permutation_only,
            )
        ),
        "disagreement_by_task": {
            task_id: {
                "pair_count": task_totals[task_id],
                "repeat_disagreement_count": (
                    task_repeat_disagreements[task_id]
                ),
                "repeat_disagreement_rate": (
                    task_repeat_disagreements[task_id]
                    / task_totals[task_id]
                ),
                "permutation_disagreement_count": (
                    task_permutation_disagreements[
                        task_id
                    ]
                ),
                "permutation_disagreement_rate": (
                    task_permutation_disagreements[
                        task_id
                    ]
                    / task_totals[task_id]
                ),
            }
            for task_id in sorted(task_totals)
        },
        "pairs": pair_results,
    }

    output_directory = (
        args.source_run
        / "experiments"
        / "instability_control_comparison"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{model_slug}-instability-control.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "pair_count": report["pair_count"],
        "repeat_disagreement_count": (
            report["repeat_disagreement_count"]
        ),
        "repeat_disagreement_rate": (
            report["repeat_disagreement_rate"]
        ),
        "permutation_disagreement_count": (
            report["permutation_disagreement_count"]
        ),
        "permutation_disagreement_rate": (
            report["permutation_disagreement_rate"]
        ),
        "absolute_excess_disagreement": (
            report["absolute_excess_disagreement"]
        ),
        "relative_disagreement_ratio": (
            report["relative_disagreement_ratio"]
        ),
        "paired_outcome_counts": (
            report["paired_outcome_counts"]
        ),
        "mcnemar_exact_two_sided_p": (
            report["mcnemar_exact_two_sided_p"]
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nSaved analysis: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
