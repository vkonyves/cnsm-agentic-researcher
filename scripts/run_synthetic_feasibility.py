from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord
from cnsm_agentic.model_io import (
    format_mcqa_prompt,
    parse_mcqa_output,
)
from cnsm_agentic.prediction_schemas import (
    PredictionRecord,
    PredictionRunReport,
)


DEFAULT_SEED = 20260726


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic end-to-end MCQA feasibility test "
            "without model or API calls."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_records(
    path: Path,
) -> list[MCQARecord]:
    records: list[MCQARecord] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                records.append(
                    MCQARecord.model_validate_json(
                        stripped
                    )
                )

            except Exception as exc:
                raise ValueError(
                    f"Invalid record at line {line_number}: {exc}"
                ) from exc

    return records


def generate_synthetic_output(
    record: MCQARecord,
    generator: random.Random,
) -> str:
    """
    Generate controlled output variants.

    Most outputs are parseable, but some are deliberately
    ambiguous, invalid, or empty.
    """

    valid_labels = sorted(
        record.options.keys()
    )

    incorrect_labels = [
        label
        for label in valid_labels
        if label != record.correct_option
    ]

    roll = generator.random()

    if roll < 0.30:
        return record.correct_option

    if roll < 0.45:
        return f"{record.correct_option}."

    if roll < 0.60:
        return (
            f"The answer is {record.correct_option}."
        )

    if roll < 0.70:
        return (
            f"I select option {record.correct_option}."
        )

    if roll < 0.78:
        wrong = generator.choice(
            incorrect_labels
        )
        return wrong

    if roll < 0.84:
        wrong = generator.choice(
            incorrect_labels
        )
        return (
            f"The answer is {wrong}."
        )

    if roll < 0.90:
        wrong = generator.choice(
            incorrect_labels
        )
        return (
            f"It could be {record.correct_option} "
            f"or {wrong}."
        )

    if roll < 0.95:
        return "E"

    return ""


def main() -> int:
    args = parse_args()

    sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Feasibility sample not found: {sample_path}"
        )

    records = load_records(
        sample_path
    )

    generator = random.Random(
        args.seed
    )

    generator_name = (
        "synthetic_mixed_output_generator_v1"
    )

    predictions: list[
        PredictionRecord
    ] = []

    status_counts: Counter[str] = Counter()
    parser_rule_counts: Counter[str] = Counter()

    task_scored: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()

    for record in records:
        prompt = format_mcqa_prompt(
            record
        )

        raw_output = generate_synthetic_output(
            record=record,
            generator=generator,
        )

        parsed = parse_mcqa_output(
            raw_output=raw_output,
            valid_options=set(
                record.options.keys()
            ),
        )

        is_correct: bool | None

        if parsed.parsed_option is None:
            is_correct = None
        else:
            is_correct = (
                parsed.parsed_option
                == record.correct_option
            )

            task_scored[
                record.task_id
            ] += 1

            if is_correct:
                task_correct[
                    record.task_id
                ] += 1

        status_counts[
            parsed.parse_status
        ] += 1

        if parsed.parser_rule:
            parser_rule_counts[
                parsed.parser_rule
            ] += 1

        predictions.append(
            PredictionRecord(
                question_id=record.question_id,
                benchmark=record.benchmark,
                task_id=record.task_id,
                gold_option=record.correct_option,
                raw_output=raw_output,
                parsed_option=parsed.parsed_option,
                parse_status=parsed.parse_status,
                parser_rule=parsed.parser_rule,
                is_correct=is_correct,
                prompt_sha256=sha256_text(
                    prompt
                ),
                generator_name=generator_name,
            )
        )

    output_directory = (
        args.source_run
        / "experiments"
        / "synthetic_feasibility"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_directory
        / "predictions.jsonl"
    )

    report_path = (
        output_directory
        / "report.json"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for prediction in predictions:
            handle.write(
                prediction.model_dump_json()
                + "\n"
            )

    scored_predictions = [
        prediction
        for prediction in predictions
        if prediction.is_correct is not None
    ]

    correct_count = sum(
        prediction.is_correct is True
        for prediction in predictions
    )

    task_ids = sorted(
        {
            record.task_id
            for record in records
        }
    )

    task_accuracy: dict[
        str,
        float | None,
    ] = {}

    for task_id in task_ids:
        if task_scored[task_id] == 0:
            task_accuracy[task_id] = None
        else:
            task_accuracy[task_id] = (
                task_correct[task_id]
                / task_scored[task_id]
            )

    scored_count = len(
        scored_predictions
    )

    report = PredictionRunReport(
        benchmark="6G-Bench",
        sample_path=str(
            sample_path.relative_to(
                args.source_run
            )
        ),
        generator_name=generator_name,
        seed=args.seed,
        record_count=len(records),
        parsed_count=status_counts["parsed"],
        ambiguous_count=status_counts[
            "ambiguous"
        ],
        unparsed_count=status_counts[
            "unparsed"
        ],
        empty_count=status_counts[
            "empty"
        ],
        scored_count=scored_count,
        correct_count=correct_count,
        accuracy_on_scored=(
            correct_count / scored_count
            if scored_count
            else None
        ),
        coverage=(
            scored_count / len(records)
            if records
            else 0.0
        ),
        parser_rule_counts=dict(
            parser_rule_counts
        ),
        parse_status_counts=dict(
            status_counts
        ),
        task_accuracy=task_accuracy,
        predictions_path=str(
            predictions_path.relative_to(
                args.source_run
            )
        ),
    )

    report_path.write_text(
        report.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    summary = {
        "record_count": report.record_count,
        "parsed_count": report.parsed_count,
        "ambiguous_count": (
            report.ambiguous_count
        ),
        "unparsed_count": (
            report.unparsed_count
        ),
        "empty_count": report.empty_count,
        "scored_count": report.scored_count,
        "correct_count": report.correct_count,
        "accuracy_on_scored": (
            report.accuracy_on_scored
        ),
        "coverage": report.coverage,
        "parse_status_counts": (
            report.parse_status_counts
        ),
        "parser_rule_counts": (
            report.parser_rule_counts
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
        f"\nSaved predictions: {predictions_path}"
    )

    print(
        f"Saved report: {report_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
