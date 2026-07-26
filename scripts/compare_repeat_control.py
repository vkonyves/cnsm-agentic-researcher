from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two predictions over the same unchanged "
            "6G-Bench sample."
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

    parser.add_argument(
        "--first-label",
        default="n60",
    )

    parser.add_argument(
        "--second-label",
        default="repeat2-n60",
    )

    return parser.parse_args()


def load_records(
    path: Path,
) -> dict[str, MCQARecord]:
    records: dict[str, MCQARecord] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = MCQARecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid MCQA record at line "
                    f"{line_number}: {exc}"
                ) from exc

            if record.question_id in records:
                raise ValueError(
                    f"Duplicate question ID: {record.question_id}"
                )

            records[record.question_id] = record

    return records


def load_predictions(
    path: Path,
) -> dict[str, dict[str, object]]:
    predictions: dict[str, dict[str, object]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            question_id = str(item["question_id"])

            if question_id in predictions:
                raise ValueError(
                    f"Duplicate prediction ID: {question_id}"
                )

            predictions[question_id] = item

    return predictions


def main() -> int:
    args = parse_args()

    model_slug = args.model.replace("/", "-")

    sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    first_path = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / f"{model_slug}-{args.first_label}"
        / "predictions.jsonl"
    )

    second_path = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / f"{model_slug}-{args.second_label}"
        / "predictions.jsonl"
    )

    for path in [
        sample_path,
        first_path,
        second_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    records = load_records(sample_path)
    first_predictions = load_predictions(first_path)
    second_predictions = load_predictions(second_path)

    expected_ids = set(records)

    if set(first_predictions) != expected_ids:
        raise ValueError(
            "First run question IDs do not match sample."
        )

    if set(second_predictions) != expected_ids:
        raise ValueError(
            "Second run question IDs do not match sample."
        )

    paired_outcomes: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()

    task_totals: Counter[str] = Counter()
    task_disagreements: Counter[str] = Counter()

    agreement_count = 0
    first_correct_count = 0
    second_correct_count = 0

    pairs: list[dict[str, object]] = []

    for question_id, record in records.items():
        first_item = first_predictions[question_id]
        second_item = second_predictions[question_id]

        first_label = first_item.get("parsed_option")
        second_label = second_item.get("parsed_option")

        if first_label is None or second_label is None:
            raise ValueError(
                f"Unparsed prediction for {question_id}"
            )

        first_label = str(first_label).upper()
        second_label = str(second_label).upper()

        first_text = record.options[first_label]
        second_text = record.options[second_label]

        semantic_agreement = first_text == second_text
        label_agreement = first_label == second_label

        # Since the sample is unchanged, these should always agree.
        if semantic_agreement != label_agreement:
            raise RuntimeError(
                f"Unexpected label/text mismatch for {question_id}"
            )

        first_correct = (
            first_label == record.correct_option
        )
        second_correct = (
            second_label == record.correct_option
        )

        first_correct_count += int(first_correct)
        second_correct_count += int(second_correct)

        task_totals[record.task_id] += 1

        if semantic_agreement:
            agreement_count += 1
        else:
            task_disagreements[record.task_id] += 1

        if first_correct and second_correct:
            outcome = "both_correct"
        elif first_correct and not second_correct:
            outcome = "first_only_correct"
        elif not first_correct and second_correct:
            outcome = "second_only_correct"
        else:
            outcome = "both_wrong"

        paired_outcomes[outcome] += 1
        transition_counts[
            f"{first_label}->{second_label}"
        ] += 1

        pairs.append(
            {
                "question_id": question_id,
                "task_id": record.task_id,
                "task_name": record.task_name,
                "gold_option": record.correct_option,
                "first_predicted_option": first_label,
                "second_predicted_option": second_label,
                "semantic_agreement": semantic_agreement,
                "first_correct": first_correct,
                "second_correct": second_correct,
                "paired_outcome": outcome,
            }
        )

    pair_count = len(pairs)
    disagreement_count = (
        pair_count - agreement_count
    )

    report = {
        "benchmark": "6G-Bench",
        "model": args.model,
        "first_run_label": args.first_label,
        "second_run_label": args.second_label,
        "pair_count": pair_count,
        "first_accuracy": (
            first_correct_count / pair_count
        ),
        "second_accuracy": (
            second_correct_count / pair_count
        ),
        "accuracy_difference": (
            second_correct_count / pair_count
            - first_correct_count / pair_count
        ),
        "semantic_agreement_count": agreement_count,
        "semantic_agreement_rate": (
            agreement_count / pair_count
        ),
        "semantic_disagreement_count": disagreement_count,
        "semantic_disagreement_rate": (
            disagreement_count / pair_count
        ),
        "paired_outcome_counts": dict(
            sorted(paired_outcomes.items())
        ),
        "label_transition_counts": dict(
            sorted(transition_counts.items())
        ),
        "disagreement_by_task": {
            task_id: {
                "pair_count": task_totals[task_id],
                "disagreement_count": (
                    task_disagreements[task_id]
                ),
                "disagreement_rate": (
                    task_disagreements[task_id]
                    / task_totals[task_id]
                ),
            }
            for task_id in sorted(task_totals)
        },
        "pairs": pairs,
    }

    output_directory = (
        args.source_run
        / "experiments"
        / "repeatability_comparison"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / (
            f"{model_slug}-"
            f"{args.first_label}-vs-"
            f"{args.second_label}.json"
        )
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
        "first_accuracy": report["first_accuracy"],
        "second_accuracy": report["second_accuracy"],
        "accuracy_difference": (
            report["accuracy_difference"]
        ),
        "semantic_agreement_count": (
            report["semantic_agreement_count"]
        ),
        "semantic_agreement_rate": (
            report["semantic_agreement_rate"]
        ),
        "semantic_disagreement_count": (
            report["semantic_disagreement_count"]
        ),
        "semantic_disagreement_rate": (
            report["semantic_disagreement_rate"]
        ),
        "paired_outcome_counts": (
            report["paired_outcome_counts"]
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nSaved comparison: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
