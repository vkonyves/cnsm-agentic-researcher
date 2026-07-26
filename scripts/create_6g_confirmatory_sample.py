from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


DEFAULT_SEED = 20260728
RECORDS_PER_TASK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a disjoint, task-stratified 300-question "
            "6G-Bench confirmatory sample."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
                record = MCQARecord.model_validate_json(
                    stripped
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid record at line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def main() -> int:
    args = parse_args()

    source_dataset_path = (
        args.source_run
        / "datasets"
        / "normalised"
        / "6g_bench_mcqa.jsonl"
    )

    pilot_sample_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    output_directory = (
        args.source_run
        / "datasets"
        / "samples"
    )

    output_path = (
        output_directory
        / "6g_bench_confirmatory_10_per_task.jsonl"
    )

    manifest_path = (
        output_directory
        / "6g_bench_confirmatory_10_per_task_manifest.json"
    )

    for path in [
        source_dataset_path,
        pilot_sample_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}"
        )

    all_records = load_records(
        source_dataset_path
    )

    pilot_records = load_records(
        pilot_sample_path
    )

    pilot_ids = {
        record.question_id
        for record in pilot_records
    }

    records_by_task: dict[
        str,
        list[MCQARecord],
    ] = defaultdict(list)

    for record in all_records:
        if record.question_id in pilot_ids:
            continue

        records_by_task[
            record.task_id
        ].append(record)

    if len(records_by_task) != 30:
        raise ValueError(
            f"Expected 30 tasks after pilot exclusion, "
            f"found {len(records_by_task)}."
        )

    insufficient_tasks = {
        task_id: len(records)
        for task_id, records
        in records_by_task.items()
        if len(records) < RECORDS_PER_TASK
    }

    if insufficient_tasks:
        raise ValueError(
            "Insufficient unseen records for tasks: "
            f"{insufficient_tasks}"
        )

    generator = random.Random(
        args.seed
    )

    selected_records: list[MCQARecord] = []
    selected_counts: Counter[str] = Counter()

    for task_id in sorted(records_by_task):
        task_records = list(
            records_by_task[task_id]
        )

        generator.shuffle(task_records)

        selected = task_records[
            :RECORDS_PER_TASK
        ]

        selected_records.extend(selected)
        selected_counts[task_id] += len(
            selected
        )

    selected_records.sort(
        key=lambda record: (
            record.task_id,
            record.question_id,
        )
    )

    selected_ids = {
        record.question_id
        for record in selected_records
    }

    overlap_with_pilot = sorted(
        selected_ids & pilot_ids
    )

    duplicate_count = (
        len(selected_records)
        - len(selected_ids)
    )

    if len(selected_records) != 300:
        raise RuntimeError(
            f"Expected 300 records, "
            f"found {len(selected_records)}."
        )

    if overlap_with_pilot:
        raise RuntimeError(
            "Confirmatory sample overlaps "
            f"with pilot: {overlap_with_pilot}"
        )

    if duplicate_count:
        raise RuntimeError(
            f"Duplicate confirmatory records: "
            f"{duplicate_count}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in selected_records:
            handle.write(
                record.model_dump_json()
                + "\n"
            )

    manifest = {
        "benchmark": "6G-Bench",
        "purpose": "confirmatory",
        "seed": args.seed,
        "source_dataset": str(
            source_dataset_path.relative_to(
                args.source_run
            )
        ),
        "source_dataset_sha256": sha256_file(
            source_dataset_path
        ),
        "pilot_sample": str(
            pilot_sample_path.relative_to(
                args.source_run
            )
        ),
        "pilot_sample_sha256": sha256_file(
            pilot_sample_path
        ),
        "record_count": len(
            selected_records
        ),
        "task_count": len(
            selected_counts
        ),
        "records_per_task": RECORDS_PER_TASK,
        "excluded_pilot_question_count": len(
            pilot_ids
        ),
        "overlap_with_pilot_count": len(
            overlap_with_pilot
        ),
        "duplicate_count": duplicate_count,
        "task_counts": dict(
            sorted(selected_counts.items())
        ),
        "selected_question_ids": [
            record.question_id
            for record in selected_records
        ],
        "output_sample": str(
            output_path.relative_to(
                args.source_run
            )
        ),
        "output_sample_sha256": sha256_file(
            output_path
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "record_count": (
            manifest["record_count"]
        ),
        "task_count": (
            manifest["task_count"]
        ),
        "records_per_task": (
            manifest["records_per_task"]
        ),
        "excluded_pilot_question_count": (
            manifest[
                "excluded_pilot_question_count"
            ]
        ),
        "overlap_with_pilot_count": (
            manifest[
                "overlap_with_pilot_count"
            ]
        ),
        "duplicate_count": (
            manifest["duplicate_count"]
        ),
        "seed": manifest["seed"],
        "output_sample_sha256": (
            manifest[
                "output_sample_sha256"
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
        f"\nSaved confirmatory sample: "
        f"{output_path}"
    )

    print(
        f"Saved manifest: "
        f"{manifest_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
