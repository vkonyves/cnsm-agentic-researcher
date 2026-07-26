from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cnsm_agentic.benchmark_schemas import (
    MCQARecord,
    NormalisationReport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert extracted 6G-Bench MCQA files "
            "into a canonical JSONL dataset."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def stable_question_id(
    task_id: str,
    source_file: str,
    question_index: int,
    question: str,
) -> str:
    payload = "|".join(
        [
            task_id,
            source_file,
            str(question_index),
            question,
        ]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"6gbench-{task_id}-{digest}"


def option_label_signature(
    options: dict[str, str],
) -> str:
    return ",".join(
        sorted(options.keys())
    )


def main() -> int:
    args = parse_args()

    source_root = (
        args.source_run
        / "inspection"
        / "extracted"
        / "6G-Bench-3k-Validated"
        / "mcq_questions_only"
    )

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source directory not found: "
            f"{source_root}"
        )

    output_directory = (
        args.source_run
        / "datasets"
        / "normalised"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_jsonl = (
        output_directory
        / "6g_bench_mcqa.jsonl"
    )

    report_path = (
        output_directory
        / "6g_bench_normalisation_report.json"
    )

    source_files = sorted(
        source_root.rglob("*.json")
    )

    question_file_count = 0
    raw_question_count = 0

    records: list[MCQARecord] = []
    invalid_records: list[dict[str, object]] = []

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    option_label_sets: Counter[str] = Counter()
    correct_option_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    rationale_tag_counts: Counter[str] = Counter()

    for source_path in source_files:
        relative_source = str(
            source_path.relative_to(
                source_root
            )
        )

        try:
            payload: Any = json.loads(
                source_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:
            invalid_records.append(
                {
                    "source_file": relative_source,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue

        if not isinstance(payload, dict):
            continue

        questions = payload.get("questions")

        if questions is None:
            continue

        question_file_count += 1

        if not isinstance(questions, list):
            invalid_records.append(
                {
                    "source_file": relative_source,
                    "error": (
                        "questions field is not a list"
                    ),
                }
            )
            continue

        for question_index, raw in enumerate(
            questions
        ):
            raw_question_count += 1

            if not isinstance(raw, dict):
                invalid_records.append(
                    {
                        "source_file": relative_source,
                        "question_index": (
                            question_index
                        ),
                        "error": (
                            "question record is not "
                            "a dictionary"
                        ),
                    }
                )
                continue

            try:
                task_id = str(
                    raw["task_id"]
                ).strip()

                question_text = str(
                    raw["question"]
                ).strip()

                question_id = stable_question_id(
                    task_id=task_id,
                    source_file=relative_source,
                    question_index=question_index,
                    question=question_text,
                )

                record = MCQARecord(
                    benchmark="6G-Bench",
                    question_id=question_id,
                    task_id=task_id,
                    task_name=str(
                        raw["task_name"]
                    ).strip(),
                    source_turn=int(
                        raw["source_turn"]
                    ),
                    question=question_text,
                    options=raw["options"],
                    correct_option=str(
                        raw["correct"]
                    ),
                    rationale=str(
                        raw["reason"]
                    ).strip(),
                    rationale_tag=str(
                        raw["rationale_tag"]
                    ).strip(),
                    difficulty=str(
                        raw["difficulty"]
                    ).strip(),
                    source_file=relative_source,
                    source_question_index=(
                        question_index
                    ),
                )

                if (
                    record.correct_option
                    not in record.options
                ):
                    raise ValueError(
                        "correct_option is not present "
                        "in options."
                    )

                if record.question_id in seen_ids:
                    duplicate_ids.append(
                        record.question_id
                    )
                    continue

                seen_ids.add(
                    record.question_id
                )

                records.append(record)

                option_label_sets[
                    option_label_signature(
                        record.options
                    )
                ] += 1

                correct_option_counts[
                    record.correct_option
                ] += 1

                task_counts[
                    record.task_id
                ] += 1

                difficulty_counts[
                    record.difficulty
                ] += 1

                rationale_tag_counts[
                    record.rationale_tag
                ] += 1

            except Exception as exc:
                invalid_records.append(
                    {
                        "source_file": relative_source,
                        "question_index": (
                            question_index
                        ),
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

    with output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                record.model_dump_json()
                + "\n"
            )

    report = NormalisationReport(
        benchmark="6G-Bench",
        source_root=str(
            source_root.relative_to(
                args.source_run
            )
        ),
        output_jsonl=str(
            output_jsonl.relative_to(
                args.source_run
            )
        ),
        source_file_count=len(
            source_files
        ),
        question_file_count=(
            question_file_count
        ),
        raw_question_count=(
            raw_question_count
        ),
        normalised_record_count=len(
            records
        ),
        duplicate_question_id_count=len(
            duplicate_ids
        ),
        invalid_record_count=len(
            invalid_records
        ),
        option_label_sets=dict(
            option_label_sets
        ),
        correct_option_counts=dict(
            correct_option_counts
        ),
        task_counts=dict(
            sorted(
                task_counts.items()
            )
        ),
        difficulty_counts=dict(
            difficulty_counts
        ),
        rationale_tag_counts=dict(
            rationale_tag_counts
        ),
        invalid_records=invalid_records,
    )

    report_path.write_text(
        report.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    concise = {
        "source_file_count": (
            report.source_file_count
        ),
        "question_file_count": (
            report.question_file_count
        ),
        "raw_question_count": (
            report.raw_question_count
        ),
        "normalised_record_count": (
            report.normalised_record_count
        ),
        "duplicate_question_id_count": (
            report.duplicate_question_id_count
        ),
        "invalid_record_count": (
            report.invalid_record_count
        ),
        "option_label_sets": (
            report.option_label_sets
        ),
        "correct_option_counts": (
            report.correct_option_counts
        ),
        "difficulty_counts": (
            report.difficulty_counts
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
        f"\nSaved dataset: {output_jsonl}"
    )

    print(
        f"Saved report: {report_path}"
    )

    return (
        0
        if (
            report.invalid_record_count == 0
            and report.duplicate_question_id_count
            == 0
            and report.normalised_record_count
            == 3722
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
