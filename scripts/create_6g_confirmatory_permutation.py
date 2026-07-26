from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from cnsm_agentic.benchmark_schemas import MCQARecord


DEFAULT_SEED = 20260729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducibly option-permuted version of the "
            "6G-Bench 300-question confirmatory sample."
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
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
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
                    f"Invalid record at line {line_number}: {exc}"
                ) from exc

            if record.question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question ID: {record.question_id}"
                )

            seen_ids.add(record.question_id)
            records.append(record)

    return records


def make_permuted_question_id(
    *,
    original_question_id: str,
    seed: int,
    old_to_new: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "original_question_id": original_question_id,
            "seed": seed,
            "old_to_new": old_to_new,
        },
        sort_keys=True,
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"{original_question_id}"
        f"-perm-{digest}"
    )


def main() -> int:
    args = parse_args()

    samples_directory = (
        args.source_run
        / "datasets"
        / "samples"
    )

    source_path = (
        samples_directory
        / "6g_bench_confirmatory_10_per_task.jsonl"
    )

    source_manifest_path = (
        samples_directory
        / "6g_bench_confirmatory_10_per_task_manifest.json"
    )

    output_path = (
        samples_directory
        / (
            "6g_bench_confirmatory_"
            "10_per_task_permuted.jsonl"
        )
    )

    manifest_path = (
        samples_directory
        / (
            "6g_bench_confirmatory_"
            "10_per_task_permuted_manifest.json"
        )
    )

    for path in [
        source_path,
        source_manifest_path,
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

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    expected_source_hash = source_manifest.get(
        "output_sample_sha256"
    )

    actual_source_hash = sha256_file(
        source_path
    )

    if (
        expected_source_hash is not None
        and expected_source_hash
        != actual_source_hash
    ):
        raise ValueError(
            "Confirmatory source sample hash does not "
            "match its manifest."
        )

    records = load_records(
        source_path
    )

    if len(records) != 300:
        raise ValueError(
            f"Expected 300 source records, "
            f"found {len(records)}."
        )

    generator = random.Random(
        args.seed
    )

    permuted_records: list[MCQARecord] = []
    manifest_items: list[dict[str, object]] = []

    original_gold_counts: Counter[str] = Counter()
    permuted_gold_counts: Counter[str] = Counter()

    identity_permutation_count = 0

    for record in records:
        original_labels = sorted(
            record.options
        )

        if original_labels != [
            "A",
            "B",
            "C",
            "D",
        ]:
            raise ValueError(
                f"Unexpected option labels for "
                f"{record.question_id}: "
                f"{original_labels}"
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

        # Map each old label to its new label.
        old_to_new = dict(
            zip(
                original_labels,
                shuffled_labels,
            )
        )

        new_options = {
            new_label: record.options[
                old_label
            ]
            for old_label, new_label
            in old_to_new.items()
        }

        new_correct_option = old_to_new[
            record.correct_option
        ]

        new_question_id = (
            make_permuted_question_id(
                original_question_id=(
                    record.question_id
                ),
                seed=args.seed,
                old_to_new=old_to_new,
            )
        )

        permuted_record = record.model_copy(
            update={
                "question_id": new_question_id,
                "options": new_options,
                "correct_option": (
                    new_correct_option
                ),
            }
        )

        permuted_record = (
            MCQARecord.model_validate(
                permuted_record.model_dump()
            )
        )

        if (
            permuted_record.correct_option
            not in permuted_record.options
        ):
            raise RuntimeError(
                "Permuted correct option is absent "
                f"for {record.question_id}."
            )

        if old_to_new == {
            label: label
            for label in original_labels
        }:
            identity_permutation_count += 1

        original_gold_counts[
            record.correct_option
        ] += 1

        permuted_gold_counts[
            permuted_record.correct_option
        ] += 1

        permuted_records.append(
            permuted_record
        )

        manifest_items.append(
            {
                "original_question_id": (
                    record.question_id
                ),
                "permuted_question_id": (
                    permuted_record.question_id
                ),
                "task_id": record.task_id,
                "old_to_new_label": (
                    old_to_new
                ),
                "original_correct_option": (
                    record.correct_option
                ),
                "permuted_correct_option": (
                    permuted_record.correct_option
                ),
            }
        )

    if identity_permutation_count != 0:
        raise RuntimeError(
            "At least one identity permutation "
            "was generated."
        )

    if len(permuted_records) != 300:
        raise RuntimeError(
            f"Expected 300 permuted records, "
            f"found {len(permuted_records)}."
        )

    original_ids = {
        record.question_id
        for record in records
    }

    permuted_ids = {
        record.question_id
        for record in permuted_records
    }

    duplicate_permuted_id_count = (
        len(permuted_records)
        - len(permuted_ids)
    )

    if duplicate_permuted_id_count:
        raise RuntimeError(
            "Duplicate permuted question IDs: "
            f"{duplicate_permuted_id_count}"
        )

    if original_ids & permuted_ids:
        raise RuntimeError(
            "Original and permuted question IDs overlap."
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
        "purpose": (
            "confirmatory option-permutation test"
        ),
        "seed": args.seed,
        "source_sample": str(
            source_path.relative_to(
                args.source_run
            )
        ),
        "source_sample_sha256": (
            actual_source_hash
        ),
        "record_count": len(
            permuted_records
        ),
        "identity_permutation_count": (
            identity_permutation_count
        ),
        "duplicate_permuted_id_count": (
            duplicate_permuted_id_count
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
        "seed": manifest["seed"],
        "identity_permutation_count": (
            manifest[
                "identity_permutation_count"
            ]
        ),
        "duplicate_permuted_id_count": (
            manifest[
                "duplicate_permuted_id_count"
            ]
        ),
        "original_gold_counts": (
            manifest["original_gold_counts"]
        ),
        "permuted_gold_counts": (
            manifest["permuted_gold_counts"]
        ),
        "source_sample_sha256": (
            manifest["source_sample_sha256"]
        ),
        "output_sample_sha256": (
            manifest["output_sample_sha256"]
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
        f"\nSaved permuted confirmatory sample: "
        f"{output_path}"
    )

    print(
        f"Saved permutation manifest: "
        f"{manifest_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
