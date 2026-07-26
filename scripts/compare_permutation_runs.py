from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and option-permuted 6G-Bench "
            "predictions question by question."
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


def load_jsonl_records(
    path: Path,
) -> dict[str, MCQARecord]:
    records: dict[str, MCQARecord] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = MCQARecord.model_validate_json(
                    stripped
                )

            except Exception as exc:
                raise ValueError(
                    f"Invalid MCQA record at line "
                    f"{line_number}: {exc}"
                ) from exc

            if record.question_id in records:
                raise ValueError(
                    f"Duplicate question ID: "
                    f"{record.question_id}"
                )

            records[record.question_id] = record

    return records


def load_predictions(
    path: Path,
) -> dict[str, dict[str, object]]:
    predictions: dict[
        str,
        dict[str, object],
    ] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                item = json.loads(stripped)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            question_id = str(item["question_id"])

            if question_id in predictions:
                raise ValueError(
                    f"Duplicate prediction ID: "
                    f"{question_id}"
                )

            predictions[question_id] = item

    return predictions


def exact_mcnemar_p_value(
    original_only_correct: int,
    permuted_only_correct: int,
) -> float | None:
    """
    Exact two-sided McNemar/binomial test.

    Only discordant pairs contribute:
    - original correct, permuted wrong
    - original wrong, permuted correct
    """

    discordant = (
        original_only_correct
        + permuted_only_correct
    )

    if discordant == 0:
        return None

    smaller = min(
        original_only_correct,
        permuted_only_correct,
    )

    cumulative = sum(
        math.comb(discordant, k)
        for k in range(
            smaller + 1
        )
    ) / (2 ** discordant)

    return min(
        1.0,
        2.0 * cumulative,
    )


def main() -> int:
    args = parse_args()

    model_slug = args.model.replace(
        "/",
        "-",
    )

    original_sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    permuted_sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task_permuted.jsonl"
    )

    manifest_path = (
        args.source_run
        / "datasets"
        / "samples"
        / (
            "6g_bench_feasibility_2_per_task_"
            "permuted_manifest.json"
        )
    )

    original_predictions_path = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / f"{model_slug}-n60"
        / "predictions.jsonl"
    )

    permuted_predictions_path = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / f"{model_slug}-permuted-n60"
        / "predictions.jsonl"
    )

    required_paths = [
        original_sample_path,
        permuted_sample_path,
        manifest_path,
        original_predictions_path,
        permuted_predictions_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    original_records = load_jsonl_records(
        original_sample_path
    )

    permuted_records = load_jsonl_records(
        permuted_sample_path
    )

    original_predictions = load_predictions(
        original_predictions_path
    )

    permuted_predictions = load_predictions(
        permuted_predictions_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    pairs: list[dict[str, object]] = []

    paired_outcome_counts: Counter[str] = Counter()
    label_transition_counts: Counter[str] = Counter()
    semantic_consistency_by_task: Counter[str] = Counter()
    task_pair_counts: Counter[str] = Counter()

    semantically_consistent_count = 0
    label_consistent_count = 0

    original_correct_count = 0
    permuted_correct_count = 0

    for item in manifest["items"]:
        original_id = str(
            item["original_question_id"]
        )

        permuted_id = str(
            item["permuted_question_id"]
        )

        original_record = original_records[
            original_id
        ]

        permuted_record = permuted_records[
            permuted_id
        ]

        original_prediction = (
            original_predictions[
                original_id
            ]
        )

        permuted_prediction = (
            permuted_predictions[
                permuted_id
            ]
        )

        original_label = original_prediction.get(
            "parsed_option"
        )

        permuted_label = permuted_prediction.get(
            "parsed_option"
        )

        if original_label is None:
            raise ValueError(
                f"Original prediction is unparsed: "
                f"{original_id}"
            )

        if permuted_label is None:
            raise ValueError(
                f"Permuted prediction is unparsed: "
                f"{permuted_id}"
            )

        original_label = str(
            original_label
        ).upper()

        permuted_label = str(
            permuted_label
        ).upper()

        original_selected_text = (
            original_record.options[
                original_label
            ]
        )

        permuted_selected_text = (
            permuted_record.options[
                permuted_label
            ]
        )

        semantic_consistency = (
            original_selected_text
            == permuted_selected_text
        )

        label_consistency = (
            original_label
            == permuted_label
        )

        original_correct = (
            original_label
            == original_record.correct_option
        )

        permuted_correct = (
            permuted_label
            == permuted_record.correct_option
        )

        original_correct_count += int(
            original_correct
        )

        permuted_correct_count += int(
            permuted_correct
        )

        if original_correct and permuted_correct:
            paired_outcome = (
                "both_correct"
            )

        elif original_correct and not permuted_correct:
            paired_outcome = (
                "original_only_correct"
            )

        elif (
            not original_correct
            and permuted_correct
        ):
            paired_outcome = (
                "permuted_only_correct"
            )

        else:
            paired_outcome = (
                "both_wrong"
            )

        paired_outcome_counts[
            paired_outcome
        ] += 1

        transition = (
            f"{original_label}"
            f"->{permuted_label}"
        )

        label_transition_counts[
            transition
        ] += 1

        if semantic_consistency:
            semantically_consistent_count += 1
            semantic_consistency_by_task[
                original_record.task_id
            ] += 1

        if label_consistency:
            label_consistent_count += 1

        task_pair_counts[
            original_record.task_id
        ] += 1

        pairs.append(
            {
                "original_question_id": (
                    original_id
                ),
                "permuted_question_id": (
                    permuted_id
                ),
                "task_id": (
                    original_record.task_id
                ),
                "task_name": (
                    original_record.task_name
                ),
                "original_gold_label": (
                    original_record.correct_option
                ),
                "permuted_gold_label": (
                    permuted_record.correct_option
                ),
                "original_predicted_label": (
                    original_label
                ),
                "permuted_predicted_label": (
                    permuted_label
                ),
                "original_selected_text": (
                    original_selected_text
                ),
                "permuted_selected_text": (
                    permuted_selected_text
                ),
                "semantic_consistency": (
                    semantic_consistency
                ),
                "label_consistency": (
                    label_consistency
                ),
                "original_correct": (
                    original_correct
                ),
                "permuted_correct": (
                    permuted_correct
                ),
                "paired_outcome": (
                    paired_outcome
                ),
                "old_to_new_label": (
                    item["old_to_new_label"]
                ),
            }
        )

    pair_count = len(pairs)

    original_only_correct = (
        paired_outcome_counts[
            "original_only_correct"
        ]
    )

    permuted_only_correct = (
        paired_outcome_counts[
            "permuted_only_correct"
        ]
    )

    report = {
        "benchmark": "6G-Bench",
        "model": args.model,
        "pair_count": pair_count,
        "original_accuracy": (
            original_correct_count
            / pair_count
        ),
        "permuted_accuracy": (
            permuted_correct_count
            / pair_count
        ),
        "accuracy_difference": (
            permuted_correct_count
            / pair_count
            - original_correct_count
            / pair_count
        ),
        "paired_outcome_counts": dict(
            paired_outcome_counts
        ),
        "mcnemar_exact_two_sided_p": (
            exact_mcnemar_p_value(
                original_only_correct,
                permuted_only_correct,
            )
        ),
        "semantic_consistency_count": (
            semantically_consistent_count
        ),
        "semantic_consistency_rate": (
            semantically_consistent_count
            / pair_count
        ),
        "label_consistency_count": (
            label_consistent_count
        ),
        "label_consistency_rate": (
            label_consistent_count
            / pair_count
        ),
        "semantic_consistency_by_task": {
            task_id: {
                "consistent_count": (
                    semantic_consistency_by_task[
                        task_id
                    ]
                ),
                "pair_count": (
                    task_pair_counts[
                        task_id
                    ]
                ),
                "consistency_rate": (
                    semantic_consistency_by_task[
                        task_id
                    ]
                    / task_pair_counts[
                        task_id
                    ]
                ),
            }
            for task_id in sorted(
                task_pair_counts
            )
        },
        "label_transition_counts": dict(
            sorted(
                label_transition_counts.items()
            )
        ),
        "pairs": pairs,
        "source_original_predictions": str(
            original_predictions_path.relative_to(
                args.source_run
            )
        ),
        "source_permuted_predictions": str(
            permuted_predictions_path.relative_to(
                args.source_run
            )
        ),
    }

    output_directory = (
        args.source_run
        / "experiments"
        / "permutation_comparison"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{model_slug}-comparison.json"
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
        "pair_count": (
            report["pair_count"]
        ),
        "original_accuracy": (
            report["original_accuracy"]
        ),
        "permuted_accuracy": (
            report["permuted_accuracy"]
        ),
        "accuracy_difference": (
            report["accuracy_difference"]
        ),
        "paired_outcome_counts": (
            report[
                "paired_outcome_counts"
            ]
        ),
        "mcnemar_exact_two_sided_p": (
            report[
                "mcnemar_exact_two_sided_p"
            ]
        ),
        "semantic_consistency_count": (
            report[
                "semantic_consistency_count"
            ]
        ),
        "semantic_consistency_rate": (
            report[
                "semantic_consistency_rate"
            ]
        ),
        "label_consistency_count": (
            report[
                "label_consistency_count"
            ]
        ),
        "label_consistency_rate": (
            report[
                "label_consistency_rate"
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
        f"\nSaved comparison: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
