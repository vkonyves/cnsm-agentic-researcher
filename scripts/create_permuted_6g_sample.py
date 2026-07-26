from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


DEFAULT_SEED = 20260727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducibly option-permuted version of the "
            "6G-Bench feasibility sample."
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
                    f"Invalid JSONL record at line "
                    f"{line_number}: {exc}"
                ) from exc

    return records


def permuted_question_id(
    original_question_id: str,
    seed: int,
    permutation: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "original_question_id": original_question_id,
            "seed": seed,
            "permutation": permutation,
        },
        sort_keys=True,
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"{original_question_id}-perm-{digest}"


def main() -> int:
    args = parse_args()

    source_path = (
        args.source_run
        / "datasets"
        / "samples"
        / "6g_bench_feasibility_2_per_task.jsonl"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source sample not found: {source_path}"
        )

    records = load_records(source_path)

    output_directory = (
        args.source_run
        / "datasets"
        / "samples"
    )

    output_path = (
        output_directory
        / "6g_bench_feasibility_2_per_task_permuted.jsonl"
    )

    manifest_path = (
        output_directory
        / "6g_bench_feasibility_2_per_task_permuted_manifest.json"
    )

    generator = random.Random(args.seed)

    permuted_records: list[MCQARecord] = []
    manifest_items: list[dict[str, object]] = []

    original_gold_counts: Counter[str] = Counter()
    permuted_gold_counts: Counter[str] = Counter()

    identity_permutation_count = 0

    for record in records:
        original_labels = sorted(
            record.options.keys()
        )

        shuffled_labels = list(
            original_labels
        )

        while True:
            generator.shuffle(
                shuffled_labels
            )

            if shuffled_labels != original_labels:
                break

        # old label -> new label
        old_to_new = dict(
            zip(
                original_labels,
                shuffled_labels,
            )
        )

        # Create options under their new labels.
        new_options = {
            new_label: record.options[old_label]
            for old_label, new_label
            in old_to_new.items()
        }

        new_correct = old_to_new[
            record.correct_option
        ]

        new_question_id = permuted_question_id(
            original_question_id=record.question_id,
            seed=args.seed,
            permutation=old_to_new,
        )

        permuted = record.model_copy(
            update={
                "question_id": new_question_id,
                "options": new_options,
                "correct_option": new_correct,
            }
        )

        # Validate the copied record after updates.
        permuted = MCQARecord.model_validate(
            permuted.model_dump()
        )

        if permuted.correct_option not in permuted.options:
            raise RuntimeError(
                "Permuted gold option is absent "
                f"for {record.question_id}."
            )

        if old_to_new == {
            label: label
            for label in original_labels
        }:
            identity_permutation_count += 1

        permuted_records.append(
            permuted
        )

        original_gold_counts[
            record.correct_option
        ] += 1

        permuted_gold_counts[
            permuted.correct_option
        ] += 1

        manifest_items.append(
            {
                "original_question_id": (
                    record.question_id
                ),
                "permuted_question_id": (
                    permuted.question_id
                ),
                "old_to_new_label": (
                    old_to_new
                ),
                "original_correct_option": (
                    record.correct_option
                ),
                "permuted_correct_option": (
                    permuted.correct_option
                ),
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in permuted_records:
            handle.write(
                record.model_dump_json()
                + "\n"
            )

    manifest = {
        "benchmark": "6G-Bench",
        "source_sample": str(
            source_path.relative_to(
                args.source_run
            )
        ),
        "output_sample": str(
            output_path.relative_to(
                args.source_run
            )
        ),
        "seed": args.seed,
        "record_count": len(
            permuted_records
        ),
        "identity_permutation_count": (
            identity_permutation_count
        ),
        "original_gold_counts": dict(
            sorted(
                original_gold_counts.items()
            )
        ),
        "permuted_gold_counts": dict(
            sorted(
                permuted_gold_counts.items()
            )
        ),
        "items": manifest_items,
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
        "record_count": (
            manifest["record_count"]
        ),
        "seed": manifest["seed"],
        "identity_permutation_count": (
            manifest[
                "identity_permutation_count"
            ]
        ),
        "original_gold_counts": (
            manifest["original_gold_counts"]
        ),
        "permuted_gold_counts": (
            manifest["permuted_gold_counts"]
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
        f"\nSaved permuted sample: {output_path}"
    )

    print(
        f"Saved manifest: {manifest_path}"
    )

    return (
        0
        if (
            len(permuted_records) == len(records)
            and identity_permutation_count == 0
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
