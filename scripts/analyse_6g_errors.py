from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse task, rationale-tag, option and question-length "
            "error patterns in a completed 6G-Bench feasibility run."
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
        "--limit",
        type=int,
        default=60,
    )

    return parser.parse_args()


def load_sample(path: Path) -> dict[str, MCQARecord]:
    records: dict[str, MCQARecord] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = MCQARecord.model_validate_json(line)

            if record.question_id in records:
                raise ValueError(
                    f"Duplicate question ID at line {line_number}: "
                    f"{record.question_id}"
                )

            records[record.question_id] = record

    return records


def load_predictions(path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSON at line {line_number}: {exc}"
                ) from exc

    return predictions


def grouped_accuracy(
    totals: Counter[str],
    correct: Counter[str],
) -> dict[str, dict[str, object]]:
    return {
        group: {
            "correct_count": correct[group],
            "record_count": totals[group],
            "accuracy": (
                correct[group] / totals[group]
                if totals[group]
                else None
            ),
        }
        for group in sorted(totals)
    }


def length_bucket(word_count: int) -> str:
    if word_count < 100:
        return "<100"
    if word_count < 150:
        return "100-149"
    if word_count < 200:
        return "150-199"
    return ">=200"


def main() -> int:
    args = parse_args()

    sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    experiment_directory = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / f"{args.model.replace('/', '-')}-n{args.limit}"
    )

    predictions_path = (
        experiment_directory
        / "predictions.jsonl"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample not found: {sample_path}"
        )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}"
        )

    sample = load_sample(sample_path)
    predictions = load_predictions(predictions_path)

    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()

    rationale_totals: Counter[str] = Counter()
    rationale_correct: Counter[str] = Counter()

    length_totals: Counter[str] = Counter()
    length_correct: Counter[str] = Counter()

    gold_totals: Counter[str] = Counter()
    gold_correct: Counter[str] = Counter()

    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    task_names: dict[str, str] = {}
    incorrect_items: list[dict[str, object]] = []

    for prediction in predictions:
        question_id = str(prediction["question_id"])

        if question_id not in sample:
            raise ValueError(
                f"Prediction not found in sample: {question_id}"
            )

        record = sample[question_id]
        predicted = prediction.get("parsed_option")

        if predicted is None:
            continue

        predicted = str(predicted).upper()
        gold = record.correct_option
        correct = predicted == gold

        task_names[record.task_id] = record.task_name

        task_totals[record.task_id] += 1
        rationale_totals[record.rationale_tag] += 1
        gold_totals[gold] += 1

        bucket = length_bucket(
            len(record.question.split())
        )
        length_totals[bucket] += 1

        confusion[gold][predicted] += 1

        if correct:
            task_correct[record.task_id] += 1
            rationale_correct[record.rationale_tag] += 1
            length_correct[bucket] += 1
            gold_correct[gold] += 1
        else:
            incorrect_items.append(
                {
                    "question_id": record.question_id,
                    "task_id": record.task_id,
                    "task_name": record.task_name,
                    "rationale_tag": record.rationale_tag,
                    "gold_option": gold,
                    "predicted_option": predicted,
                    "question_word_count": len(
                        record.question.split()
                    ),
                    "source_turn": record.source_turn,
                    "question": record.question,
                }
            )

    accuracy_by_task = grouped_accuracy(
        task_totals,
        task_correct,
    )

    for task_id, values in accuracy_by_task.items():
        values["task_name"] = task_names.get(task_id)

    report = {
        "benchmark": "6G-Bench",
        "model": args.model,
        "record_count": len(predictions),
        "accuracy_by_task": accuracy_by_task,
        "accuracy_by_rationale_tag": grouped_accuracy(
            rationale_totals,
            rationale_correct,
        ),
        "accuracy_by_question_length": grouped_accuracy(
            length_totals,
            length_correct,
        ),
        "accuracy_by_gold_option": grouped_accuracy(
            gold_totals,
            gold_correct,
        ),
        "confusion_matrix": {
            gold: {
                predicted: confusion[gold][predicted]
                for predicted in sorted(
                    set(gold_totals)
                    | {
                        value
                        for row in confusion.values()
                        for value in row
                    }
                )
            }
            for gold in sorted(gold_totals)
        },
        "incorrect_count": len(incorrect_items),
        "incorrect_items": sorted(
            incorrect_items,
            key=lambda item: (
                item["task_id"],
                item["question_id"],
            ),
        ),
    }

    output_path = (
        experiment_directory
        / "error_analysis.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zero_accuracy_tasks = {
        task_id: values
        for task_id, values
        in accuracy_by_task.items()
        if values["accuracy"] == 0.0
    }

    perfect_tasks = {
        task_id: values
        for task_id, values
        in accuracy_by_task.items()
        if values["accuracy"] == 1.0
    }

    summary = {
        "incorrect_count": report["incorrect_count"],
        "zero_accuracy_tasks": zero_accuracy_tasks,
        "perfect_accuracy_task_count": len(perfect_tasks),
        "accuracy_by_rationale_tag": (
            report["accuracy_by_rationale_tag"]
        ),
        "accuracy_by_question_length": (
            report["accuracy_by_question_length"]
        ),
        "accuracy_by_gold_option": (
            report["accuracy_by_gold_option"]
        ),
        "confusion_matrix": report["confusion_matrix"],
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nSaved error analysis: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
