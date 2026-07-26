from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect 6G-Bench files containing question collections."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def type_name(value: Any) -> str:
    return type(value).__name__


def main() -> int:
    args = parse_args()

    root = (
        args.source_run
        / "inspection"
        / "extracted"
        / "6G-Bench-3k-Validated"
        / "mcq_questions_only"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"6G-Bench directory not found: {root}"
        )

    json_files = sorted(
        root.rglob("*.json")
    )

    question_files: list[Path] = []
    parse_failures: list[dict[str, str]] = []

    total_questions = 0

    question_count_distribution: Counter[int] = Counter()
    question_field_presence: Counter[str] = Counter()
    question_field_types: dict[str, Counter[str]] = {}

    example_questions: list[dict[str, Any]] = []

    malformed_question_files: list[dict[str, Any]] = []
    malformed_questions: list[dict[str, Any]] = []

    for path in json_files:
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:
            parse_failures.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if not isinstance(payload, dict):
            continue

        if "questions" not in payload:
            continue

        question_files.append(path)

        questions = payload["questions"]

        if not isinstance(questions, list):
            malformed_question_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "questions_type": type_name(questions),
                }
            )
            continue

        question_count_distribution[len(questions)] += 1
        total_questions += len(questions)

        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                malformed_questions.append(
                    {
                        "path": str(path.relative_to(root)),
                        "question_index": index,
                        "question_type": type_name(question),
                    }
                )
                continue

            for key, value in question.items():
                key_string = str(key)

                question_field_presence[key_string] += 1

                if key_string not in question_field_types:
                    question_field_types[key_string] = Counter()

                question_field_types[key_string][
                    type_name(value)
                ] += 1

            if len(example_questions) < 5:
                example_questions.append(
                    {
                        "source_file": str(
                            path.relative_to(root)
                        ),
                        "question_index": index,
                        "question": question,
                    }
                )

    summary = {
        "root": str(
            root.relative_to(args.source_run)
        ),
        "json_file_count": len(json_files),
        "question_file_count": len(question_files),
        "total_question_count": total_questions,
        "question_count_distribution": {
            str(count): frequency
            for count, frequency
            in sorted(
                question_count_distribution.items()
            )
        },
        "question_field_presence": dict(
            question_field_presence.most_common()
        ),
        "question_field_types": {
            field: dict(type_counts)
            for field, type_counts
            in sorted(
                question_field_types.items()
            )
        },
        "parse_failure_count": len(parse_failures),
        "malformed_question_file_count": len(
            malformed_question_files
        ),
        "malformed_question_count": len(
            malformed_questions
        ),
        "parse_failures": parse_failures,
        "malformed_question_files": (
            malformed_question_files
        ),
        "malformed_questions": malformed_questions,
        "example_questions": example_questions,
    }

    output_path = (
        args.source_run
        / "inspection"
        / "6g_question_summary.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    concise = {
        "json_file_count": summary["json_file_count"],
        "question_file_count": (
            summary["question_file_count"]
        ),
        "total_question_count": (
            summary["total_question_count"]
        ),
        "question_count_distribution": (
            summary["question_count_distribution"]
        ),
        "question_field_presence": (
            summary["question_field_presence"]
        ),
        "question_field_types": (
            summary["question_field_types"]
        ),
        "parse_failure_count": (
            summary["parse_failure_count"]
        ),
        "malformed_question_file_count": (
            summary["malformed_question_file_count"]
        ),
        "malformed_question_count": (
            summary["malformed_question_count"]
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
        f"\nSaved summary: {output_path}"
    )

    return (
        0
        if (
            not parse_failures
            and not malformed_question_files
            and not malformed_questions
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
