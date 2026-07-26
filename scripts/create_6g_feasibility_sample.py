from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


RANDOM_SEED = 20260726
QUESTIONS_PER_TASK = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible task-stratified feasibility "
            "sample from normalised 6G-Bench MCQA data."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--questions-per-task",
        type=int,
        default=QUESTIONS_PER_TASK,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser.parse_args()


def load_records(
    dataset_path: Path,
) -> list[MCQARecord]:
    records: list[MCQARecord] = []

    with dataset_path.open(
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
                record = (
                    MCQARecord
                    .model_validate_json(
                        stripped
                    )
                )

            except Exception as exc:
                raise ValueError(
                    "Invalid JSONL record at "
                    f"line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def main() -> int:
    args = parse_args()

    if args.questions_per_task < 1:
        raise ValueError(
            "--questions-per-task must be at least 1."
        )

    dataset_path = (
        args.source_run
        / "datasets"
        / "normalised"
        / "6g_bench_mcqa.jsonl"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Normalised dataset not found: "
            f"{dataset_path}"
        )

    records = load_records(dataset_path)

    records_by_task: dict[
        str,
        list[MCQARecord],
    ] = defaultdict(list)

    for record in records:
        records_by_task[
            record.task_id
        ].append(record)

    generator = random.Random(
        args.seed
    )

    selected_records: list[
        MCQARecord
    ] = []

    task_selection_counts: Counter[str] = (
        Counter()
    )

    insufficient_tasks: list[
        dict[str, object]
    ] = []

    for task_id in sorted(
        records_by_task
    ):
        task_records = (
            records_by_task[task_id]
        )

        if (
            len(task_records)
            < args.questions_per_task
        ):
            insufficient_tasks.append(
                {
                    "task_id": task_id,
                    "available": len(
                        task_records
                    ),
                    "requested": (
                        args.questions_per_task
                    ),
                }
            )

            selected = list(
                task_records
            )

        else:
            selected = generator.sample(
                task_records,
                args.questions_per_task,
            )

        selected = sorted(
            selected,
            key=lambda record: (
                record.question_id
            ),
        )

        selected_records.extend(
            selected
        )

        task_selection_counts[
            task_id
        ] += len(selected)

    selected_records = sorted(
        selected_records,
        key=lambda record: (
            record.task_id,
            record.question_id,
        ),
    )

    selected_ids = [
        record.question_id
        for record in selected_records
    ]

    if len(selected_ids) != len(
        set(selected_ids)
    ):
        raise RuntimeError(
            "Duplicate question IDs were selected."
        )

    output_directory = (
        args.source_run
        / "datasets"
        / "samples"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_name = (
        "6g_bench_feasibility_"
        f"{args.questions_per_task}"
        "_per_task"
    )

    output_jsonl = (
        output_directory
        / f"{sample_name}.jsonl"
    )

    manifest_path = (
        output_directory
        / f"{sample_name}_manifest.json"
    )

    with output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in selected_records:
            handle.write(
                record.model_dump_json()
                + "\n"
            )

    correct_option_counts = Counter(
        record.correct_option
        for record in selected_records
    )

    rationale_tag_counts = Counter(
        record.rationale_tag
        for record in selected_records
    )

    manifest = {
        "benchmark": "6G-Bench",
        "source_dataset": str(
            dataset_path.relative_to(
                args.source_run
            )
        ),
        "output_sample": str(
            output_jsonl.relative_to(
                args.source_run
            )
        ),
        "seed": args.seed,
        "questions_per_task": (
            args.questions_per_task
        ),
        "source_record_count": len(
            records
        ),
        "task_count": len(
            records_by_task
        ),
        "selected_record_count": len(
            selected_records
        ),
        "task_selection_counts": dict(
            sorted(
                task_selection_counts.items()
            )
        ),
        "correct_option_counts": dict(
            sorted(
                correct_option_counts.items()
            )
        ),
        "rationale_tag_counts": dict(
            sorted(
                rationale_tag_counts.items()
            )
        ),
        "insufficient_tasks": (
            insufficient_tasks
        ),
        "selected_question_ids": (
            selected_ids
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "source_record_count": (
            manifest[
                "source_record_count"
            ]
        ),
        "task_count": (
            manifest["task_count"]
        ),
        "selected_record_count": (
            manifest[
                "selected_record_count"
            ]
        ),
        "questions_per_task": (
            manifest[
                "questions_per_task"
            ]
        ),
        "seed": manifest["seed"],
        "correct_option_counts": (
            manifest[
                "correct_option_counts"
            ]
        ),
        "insufficient_task_count": len(
            insufficient_tasks
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
        f"\nSaved sample: {output_jsonl}"
    )

    print(
        f"Saved manifest: {manifest_path}"
    )

    return (
        0
        if (
            not insufficient_tasks
            and len(selected_records)
            == len(records_by_task)
            * args.questions_per_task
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
