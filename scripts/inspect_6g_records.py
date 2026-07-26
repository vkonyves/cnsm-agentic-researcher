from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MAX_FILES_TO_SAMPLE_PER_SCHEMA = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect extracted 6G-Bench JSON files and summarise "
            "their record schemas."
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


def schema_signature(payload: Any) -> str:
    if isinstance(payload, dict):
        parts = [
            f"{key}:{type_name(value)}"
            for key, value in sorted(payload.items())
        ]
        return "dict{" + ",".join(parts) + "}"

    if isinstance(payload, list):
        if not payload:
            return "list[empty]"

        return f"list[{type_name(payload[0])}]"

    return type_name(payload)


def main() -> int:
    args = parse_args()

    extraction_root = (
        args.source_run
        / "inspection"
        / "extracted"
        / "6G-Bench-3k-Validated"
    )

    if not extraction_root.exists():
        raise FileNotFoundError(
            f"Extraction directory not found: {extraction_root}"
        )

    json_files = sorted(
        extraction_root.rglob("*.json")
    )

    schema_counts: Counter[str] = Counter()
    schema_examples: dict[str, list[str]] = defaultdict(list)
    directory_counts: Counter[str] = Counter()
    parse_failures: list[dict[str, str]] = []

    candidate_field_counts: Counter[str] = Counter()
    top_level_type_counts: Counter[str] = Counter()

    for path in json_files:
        relative_path = path.relative_to(
            extraction_root
        )

        parent_group = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "<root>"
        )

        directory_counts[parent_group] += 1

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:
            parse_failures.append(
                {
                    "path": str(relative_path),
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue

        top_level_type_counts[
            type_name(payload)
        ] += 1

        signature = schema_signature(payload)
        schema_counts[signature] += 1

        if (
            len(schema_examples[signature])
            < MAX_FILES_TO_SAMPLE_PER_SCHEMA
        ):
            schema_examples[signature].append(
                str(relative_path)
            )

        if isinstance(payload, dict):
            for key in payload:
                candidate_field_counts[
                    str(key)
                ] += 1

    summary = {
        "extraction_root": str(
            extraction_root.relative_to(
                args.source_run
            )
        ),
        "json_file_count": len(json_files),
        "parsed_count": (
            len(json_files)
            - len(parse_failures)
        ),
        "parse_failure_count": len(
            parse_failures
        ),
        "top_level_type_counts": dict(
            top_level_type_counts
        ),
        "top_directory_counts": dict(
            directory_counts.most_common()
        ),
        "schema_counts": [
            {
                "schema": schema,
                "count": count,
                "example_files": (
                    schema_examples[schema]
                ),
            }
            for schema, count
            in schema_counts.most_common()
        ],
        "field_presence_counts": dict(
            candidate_field_counts.most_common()
        ),
        "parse_failures": parse_failures,
    }

    output_path = (
        args.source_run
        / "inspection"
        / "6g_record_schema_summary.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "json_file_count": (
                    summary["json_file_count"]
                ),
                "parsed_count": (
                    summary["parsed_count"]
                ),
                "parse_failure_count": (
                    summary[
                        "parse_failure_count"
                    ]
                ),
                "top_level_type_counts": (
                    summary[
                        "top_level_type_counts"
                    ]
                ),
                "top_directories": dict(
                    list(
                        summary[
                            "top_directory_counts"
                        ].items()
                    )[:10]
                ),
                "top_fields": dict(
                    list(
                        summary[
                            "field_presence_counts"
                        ].items()
                    )[:20]
                ),
                "schema_variant_count": len(
                    summary["schema_counts"]
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved summary: {output_path}"
    )

    return (
        0
        if not parse_failures
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
