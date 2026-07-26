from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cnsm_agentic.baseline_schemas import (
    BaselineReport,
    BaselineResult,
)
from cnsm_agentic.benchmark_schemas import MCQARecord


RANDOM_SEED = 20260726


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic random-choice and majority-choice "
            "baselines on normalised 6G-Bench MCQA data."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def load_records(path: Path) -> list[MCQARecord]:
    records: list[MCQARecord] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                records.append(
                    MCQARecord.model_validate_json(stripped)
                )

            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: {exc}"
                ) from exc

    return records


def evaluate_predictions(
    records: list[MCQARecord],
    predictions: list[str],
    baseline_name: str,
) -> BaselineResult:
    if len(records) != len(predictions):
        raise ValueError(
            "Record and prediction counts do not match."
        )

    correct_count = 0

    prediction_counts: Counter[str] = Counter()
    correct_option_counts: Counter[str] = Counter()

    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()

    for record, prediction in zip(records, predictions):
        prediction = prediction.strip().upper()

        prediction_counts[prediction] += 1
        correct_option_counts[record.correct_option] += 1

        task_totals[record.task_id] += 1

        if prediction == record.correct_option:
            correct_count += 1
            task_correct[record.task_id] += 1

    per_task_accuracy = {
        task_id: (
            task_correct[task_id]
            / task_totals[task_id]
        )
        for task_id in sorted(task_totals)
    }

    return BaselineResult(
        baseline_name=baseline_name,
        benchmark="6G-Bench",
        record_count=len(records),
        correct_count=correct_count,
        accuracy=correct_count / len(records),
        prediction_counts=dict(prediction_counts),
        correct_option_counts=dict(correct_option_counts),
        per_task_accuracy=per_task_accuracy,
    )


def majority_predictions(
    records: list[MCQARecord],
) -> tuple[list[str], str]:
    correct_counts = Counter(
        record.correct_option
        for record in records
    )

    majority_option = sorted(
        correct_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[0][0]

    return (
        [majority_option] * len(records),
        majority_option,
    )


def random_predictions(
    records: list[MCQARecord],
    seed: int,
) -> list[str]:
    generator = random.Random(seed)

    predictions: list[str] = []

    for record in records:
        option_labels = sorted(record.options.keys())

        predictions.append(
            generator.choice(option_labels)
        )

    return predictions


def main() -> int:
    args = parse_args()

    dataset_path = (
        args.source_run
        / "datasets"
        / "normalised"
        / "6g_bench_mcqa.jsonl"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Normalised dataset not found: {dataset_path}"
        )

    records = load_records(dataset_path)

    if len(records) != 3722:
        raise RuntimeError(
            f"Expected 3722 records, found {len(records)}."
        )

    majority_pred, majority_option = majority_predictions(
        records
    )

    random_pred = random_predictions(
        records=records,
        seed=RANDOM_SEED,
    )

    majority_result = evaluate_predictions(
        records=records,
        predictions=majority_pred,
        baseline_name=(
            f"majority_choice_{majority_option}"
        ),
    )

    random_result = evaluate_predictions(
        records=records,
        predictions=random_pred,
        baseline_name=(
            f"uniform_random_seed_{RANDOM_SEED}"
        ),
    )

    report = BaselineReport(
        benchmark="6G-Bench",
        dataset_path=str(
            dataset_path.relative_to(args.source_run)
        ),
        results=[
            majority_result,
            random_result,
        ],
    )

    output_directory = (
        args.source_run
        / "experiments"
        / "baselines"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "6g_bench_baseline_report.json"
    )

    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    summary = {
        result.baseline_name: {
            "correct_count": result.correct_count,
            "record_count": result.record_count,
            "accuracy": result.accuracy,
        }
        for result in report.results
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nSaved report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
