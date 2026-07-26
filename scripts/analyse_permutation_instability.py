from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse semantic instability between original and "
            "option-permuted MCQA runs."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
    )

    return parser.parse_args()


def grouped_summary(
    totals: Counter[str],
    unstable: Counter[str],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "pair_count": totals[key],
            "unstable_count": unstable[key],
            "instability_rate": (
                unstable[key] / totals[key]
                if totals[key]
                else 0.0
            ),
        }
        for key in sorted(totals)
    }


def main() -> int:
    args = parse_args()

    model_slug = args.model.replace("/", "-")

    comparison_path = (
        args.source_run
        / "experiments"
        / "permutation_comparison"
        / f"{model_slug}-comparison.json"
    )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {comparison_path}"
        )

    comparison = json.loads(
        comparison_path.read_text(
            encoding="utf-8"
        )
    )

    pairs = comparison["pairs"]

    task_totals: Counter[str] = Counter()
    task_unstable: Counter[str] = Counter()

    original_gold_totals: Counter[str] = Counter()
    original_gold_unstable: Counter[str] = Counter()

    permuted_gold_totals: Counter[str] = Counter()
    permuted_gold_unstable: Counter[str] = Counter()

    original_prediction_totals: Counter[str] = Counter()
    original_prediction_unstable: Counter[str] = Counter()

    outcome_totals: Counter[str] = Counter()
    outcome_unstable: Counter[str] = Counter()

    changed_answer_outcomes: Counter[str] = Counter()
    semantic_transition_counts: Counter[str] = Counter()

    unstable_pairs: list[dict[str, object]] = []

    for pair in pairs:
        task_id = str(pair["task_id"])
        original_gold = str(
            pair["original_gold_label"]
        )
        permuted_gold = str(
            pair["permuted_gold_label"]
        )
        original_prediction = str(
            pair["original_predicted_label"]
        )
        paired_outcome = str(
            pair["paired_outcome"]
        )

        stable = bool(
            pair["semantic_consistency"]
        )

        task_totals[task_id] += 1
        original_gold_totals[original_gold] += 1
        permuted_gold_totals[permuted_gold] += 1
        original_prediction_totals[
            original_prediction
        ] += 1
        outcome_totals[paired_outcome] += 1

        if stable:
            continue

        task_unstable[task_id] += 1
        original_gold_unstable[
            original_gold
        ] += 1
        permuted_gold_unstable[
            permuted_gold
        ] += 1
        original_prediction_unstable[
            original_prediction
        ] += 1
        outcome_unstable[
            paired_outcome
        ] += 1

        if (
            pair["original_correct"]
            and not pair["permuted_correct"]
        ):
            changed_outcome = (
                "changed_from_correct_to_wrong"
            )

        elif (
            not pair["original_correct"]
            and pair["permuted_correct"]
        ):
            changed_outcome = (
                "changed_from_wrong_to_correct"
            )

        elif (
            not pair["original_correct"]
            and not pair["permuted_correct"]
        ):
            changed_outcome = (
                "changed_between_wrong_answers"
            )

        else:
            # Both correct necessarily select the same semantic
            # option, so this case should never be unstable.
            changed_outcome = (
                "unexpected_both_correct_instability"
            )

        changed_answer_outcomes[
            changed_outcome
        ] += 1

        transition = (
            f"{pair['original_predicted_label']}"
            f"->{pair['permuted_predicted_label']}"
        )

        semantic_transition_counts[
            transition
        ] += 1

        unstable_pairs.append(
            {
                "task_id": pair["task_id"],
                "task_name": pair["task_name"],
                "original_question_id": (
                    pair["original_question_id"]
                ),
                "original_gold_label": (
                    pair["original_gold_label"]
                ),
                "permuted_gold_label": (
                    pair["permuted_gold_label"]
                ),
                "original_predicted_label": (
                    pair[
                        "original_predicted_label"
                    ]
                ),
                "permuted_predicted_label": (
                    pair[
                        "permuted_predicted_label"
                    ]
                ),
                "original_correct": (
                    pair["original_correct"]
                ),
                "permuted_correct": (
                    pair["permuted_correct"]
                ),
                "changed_outcome": changed_outcome,
                "original_selected_text": (
                    pair["original_selected_text"]
                ),
                "permuted_selected_text": (
                    pair["permuted_selected_text"]
                ),
                "old_to_new_label": (
                    pair["old_to_new_label"]
                ),
            }
        )

    report = {
        "benchmark": comparison["benchmark"],
        "model": comparison["model"],
        "pair_count": len(pairs),
        "semantically_stable_count": (
            len(pairs) - len(unstable_pairs)
        ),
        "semantically_unstable_count": len(
            unstable_pairs
        ),
        "semantic_instability_rate": (
            len(unstable_pairs) / len(pairs)
            if pairs
            else 0.0
        ),
        "changed_answer_outcomes": dict(
            sorted(
                changed_answer_outcomes.items()
            )
        ),
        "instability_by_task": grouped_summary(
            task_totals,
            task_unstable,
        ),
        "instability_by_original_gold": (
            grouped_summary(
                original_gold_totals,
                original_gold_unstable,
            )
        ),
        "instability_by_permuted_gold": (
            grouped_summary(
                permuted_gold_totals,
                permuted_gold_unstable,
            )
        ),
        "instability_by_original_prediction": (
            grouped_summary(
                original_prediction_totals,
                original_prediction_unstable,
            )
        ),
        "instability_by_paired_outcome": (
            grouped_summary(
                outcome_totals,
                outcome_unstable,
            )
        ),
        "label_transition_counts_for_unstable_pairs": (
            dict(
                sorted(
                    semantic_transition_counts.items()
                )
            )
        ),
        "unstable_pairs": unstable_pairs,
    }

    output_path = (
        args.source_run
        / "experiments"
        / "permutation_comparison"
        / f"{model_slug}-instability.json"
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
        "semantically_unstable_count": (
            report[
                "semantically_unstable_count"
            ]
        ),
        "semantic_instability_rate": (
            report[
                "semantic_instability_rate"
            ]
        ),
        "changed_answer_outcomes": (
            report[
                "changed_answer_outcomes"
            ]
        ),
        "instability_by_original_gold": (
            report[
                "instability_by_original_gold"
            ]
        ),
        "instability_by_original_prediction": (
            report[
                "instability_by_original_prediction"
            ]
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved instability analysis: "
        f"{output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
