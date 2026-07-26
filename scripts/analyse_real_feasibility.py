from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from cnsm_agentic.benchmark_schemas import MCQARecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse a completed real-model MCQA feasibility run."
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

    parser.add_argument(
        "--limit",
        type=int,
        default=60,
    )

    return parser.parse_args()


def load_sample(
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
                    f"Invalid sample record at line "
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
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                prediction = json.loads(stripped)
            except Exception as exc:
                raise ValueError(
                    f"Invalid prediction JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            predictions.append(prediction)

    return predictions


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None

    proportion = successes / total
    denominator = 1.0 + z**2 / total

    centre = (
        proportion
        + z**2 / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion
            * (1.0 - proportion)
            / total
            + z**2
            / (4.0 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def percentile(
    values: list[float],
    quantile: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * quantile

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def accuracy_group(
    correct: Counter[str],
    totals: Counter[str],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}

    for group in sorted(totals):
        total = totals[group]
        correct_count = correct[group]

        output[group] = {
            "correct_count": correct_count,
            "record_count": total,
            "accuracy": (
                correct_count / total
                if total
                else None
            ),
        }

    return output


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
    predictions = load_predictions(
        predictions_path
    )

    seen_prediction_ids: set[str] = set()

    gold_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()

    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()

    task_name_by_id: dict[str, str] = {}

    gold_option_totals: Counter[str] = Counter()
    gold_option_correct: Counter[str] = Counter()

    prediction_option_totals: Counter[str] = Counter()
    prediction_option_correct: Counter[str] = Counter()

    confusion: dict[str, Counter[str]] = defaultdict(
        Counter
    )

    latency_values: list[float] = []
    input_token_values: list[int] = []
    output_token_values: list[int] = []
    total_token_values: list[int] = []

    errors: list[dict[str, object]] = []
    incorrect_items: list[dict[str, object]] = []

    parsed_count = 0
    correct_count = 0

    for prediction in predictions:
        question_id = str(
            prediction["question_id"]
        )

        if question_id in seen_prediction_ids:
            raise ValueError(
                f"Duplicate prediction ID: {question_id}"
            )

        seen_prediction_ids.add(question_id)

        if question_id not in sample:
            raise ValueError(
                f"Prediction question not found "
                f"in sample: {question_id}"
            )

        record = sample[question_id]

        task_name_by_id[
            record.task_id
        ] = record.task_name

        gold = record.correct_option
        parsed = prediction.get(
            "parsed_option"
        )

        gold_counts[gold] += 1
        task_totals[record.task_id] += 1
        gold_option_totals[gold] += 1

        latency = prediction.get(
            "latency_seconds"
        )

        if isinstance(latency, (int, float)):
            latency_values.append(
                float(latency)
            )

        usage = prediction.get("usage")

        if isinstance(usage, dict):
            input_tokens = usage.get(
                "input_tokens"
            )
            output_tokens = usage.get(
                "output_tokens"
            )
            total_tokens = usage.get(
                "total_tokens"
            )

            if isinstance(input_tokens, int):
                input_token_values.append(
                    input_tokens
                )

            if isinstance(output_tokens, int):
                output_token_values.append(
                    output_tokens
                )

            if isinstance(total_tokens, int):
                total_token_values.append(
                    total_tokens
                )

        if parsed is None:
            errors.append(
                {
                    "question_id": question_id,
                    "task_id": record.task_id,
                    "task_name": record.task_name,
                    "parse_status": prediction.get(
                        "parse_status"
                    ),
                    "error_type": prediction.get(
                        "error_type"
                    ),
                    "error": prediction.get(
                        "error"
                    ),
                }
            )
            continue

        parsed = str(parsed).upper()

        parsed_count += 1
        prediction_counts[parsed] += 1
        prediction_option_totals[
            parsed
        ] += 1

        confusion[gold][parsed] += 1

        is_correct = parsed == gold

        if is_correct:
            correct_count += 1
            task_correct[record.task_id] += 1
            gold_option_correct[gold] += 1
            prediction_option_correct[
                parsed
            ] += 1
        else:
            incorrect_items.append(
                {
                    "question_id": question_id,
                    "task_id": record.task_id,
                    "task_name": record.task_name,
                    "gold_option": gold,
                    "predicted_option": parsed,
                    "rationale_tag": (
                        record.rationale_tag
                    ),
                    "source_turn": (
                        record.source_turn
                    ),
                    "question": record.question,
                }
            )

    record_count = len(predictions)

    confidence_low, confidence_high = (
        wilson_interval(
            correct_count,
            parsed_count,
        )
    )

    majority_count = (
        max(gold_counts.values())
        if gold_counts
        else 0
    )

    majority_labels = sorted(
        label
        for label, count in gold_counts.items()
        if count == majority_count
    )

    majority_accuracy = (
        majority_count / record_count
        if record_count
        else None
    )

    report = {
        "benchmark": "6G-Bench",
        "model": args.model,
        "record_count": record_count,
        "parsed_count": parsed_count,
        "coverage": (
            parsed_count / record_count
            if record_count
            else 0.0
        ),
        "correct_count": correct_count,
        "accuracy": (
            correct_count / parsed_count
            if parsed_count
            else None
        ),
        "accuracy_wilson_95": {
            "lower": confidence_low,
            "upper": confidence_high,
        },
        "sample_baselines": {
            "uniform_random_expected_accuracy": 0.25,
            "majority_labels": majority_labels,
            "majority_correct_count": majority_count,
            "majority_accuracy": majority_accuracy,
        },
        "gold_option_counts": dict(
            sorted(gold_counts.items())
        ),
        "prediction_option_counts": dict(
            sorted(
                prediction_counts.items()
            )
        ),
        "confusion_matrix": {
            gold: {
                predicted: confusion[gold][
                    predicted
                ]
                for predicted
                in sorted(
                    set(gold_counts)
                    | set(prediction_counts)
                )
            }
            for gold in sorted(
                set(gold_counts)
                | set(prediction_counts)
            )
        },
        "accuracy_by_gold_option": (
            accuracy_group(
                gold_option_correct,
                gold_option_totals,
            )
        ),
        "accuracy_by_predicted_option": (
            accuracy_group(
                prediction_option_correct,
                prediction_option_totals,
            )
        ),
        "accuracy_by_task": {
            task_id: {
                **values,
                "task_name": (
                    task_name_by_id.get(
                        task_id
                    )
                ),
            }
            for task_id, values
            in accuracy_group(
                task_correct,
                task_totals,
            ).items()
        },
        "latency_seconds": {
            "count": len(latency_values),
            "mean": (
                mean(latency_values)
                if latency_values
                else None
            ),
            "median": (
                median(latency_values)
                if latency_values
                else None
            ),
            "p90": percentile(
                latency_values,
                0.90,
            ),
            "p95": percentile(
                latency_values,
                0.95,
            ),
            "minimum": (
                min(latency_values)
                if latency_values
                else None
            ),
            "maximum": (
                max(latency_values)
                if latency_values
                else None
            ),
        },
        "token_usage": {
            "input_total": sum(
                input_token_values
            ),
            "input_mean": (
                mean(input_token_values)
                if input_token_values
                else None
            ),
            "output_total": sum(
                output_token_values
            ),
            "output_mean": (
                mean(output_token_values)
                if output_token_values
                else None
            ),
            "total": sum(
                total_token_values
            ),
            "mean_per_question": (
                mean(total_token_values)
                if total_token_values
                else None
            ),
        },
        "error_count": len(errors),
        "errors": errors,
        "incorrect_count": len(
            incorrect_items
        ),
        "incorrect_items": incorrect_items,
        "source_predictions": str(
            predictions_path.relative_to(
                args.source_run
            )
        ),
    }

    output_path = (
        experiment_directory
        / "analysis.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    concise = {
        "record_count": (
            report["record_count"]
        ),
        "correct_count": (
            report["correct_count"]
        ),
        "accuracy": report["accuracy"],
        "accuracy_wilson_95": (
            report["accuracy_wilson_95"]
        ),
        "coverage": report["coverage"],
        "sample_baselines": (
            report["sample_baselines"]
        ),
        "gold_option_counts": (
            report["gold_option_counts"]
        ),
        "prediction_option_counts": (
            report[
                "prediction_option_counts"
            ]
        ),
        "latency_seconds": (
            report["latency_seconds"]
        ),
        "token_usage": (
            report["token_usage"]
        ),
        "error_count": (
            report["error_count"]
        ),
        "incorrect_count": (
            report["incorrect_count"]
        ),
    }

    print(
        json.dumps(
            concise,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved analysis: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
