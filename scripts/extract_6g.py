from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


MAX_MEMBERS = 20_000
MAX_MEMBER_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 300 * 1024 * 1024


def safe_member_path(name: str) -> bool:
    path = Path(name)

    return (
        not path.is_absolute()
        and ".." not in path.parts
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely extract JSON files from the validated "
            "6G-Bench archive."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    archive_path = (
        args.source_run
        / "acquisition"
        / "files"
        / "6G-Bench"
        / "6GBench_3k_Validated.zip"
    )

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}"
        )

    output_root = (
        args.source_run
        / "inspection"
        / "extracted"
        / "6G-Bench-3k-Validated"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_files: list[str] = []
    skipped_files: list[dict[str, str]] = []
    total_bytes = 0

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()

        if len(members) > MAX_MEMBERS:
            raise RuntimeError(
                f"Archive contains {len(members)} members, "
                f"above safety limit {MAX_MEMBERS}."
            )

        for member in members:
            if member.is_dir():
                continue

            if not safe_member_path(member.filename):
                skipped_files.append(
                    {
                        "name": member.filename,
                        "reason": "unsafe path",
                    }
                )
                continue

            if Path(member.filename).suffix.lower() != ".json":
                skipped_files.append(
                    {
                        "name": member.filename,
                        "reason": "not JSON",
                    }
                )
                continue

            if member.file_size > MAX_MEMBER_BYTES:
                skipped_files.append(
                    {
                        "name": member.filename,
                        "reason": "member exceeds size limit",
                    }
                )
                continue

            if (
                total_bytes + member.file_size
                > MAX_TOTAL_BYTES
            ):
                skipped_files.append(
                    {
                        "name": member.filename,
                        "reason": "total extraction limit reached",
                    }
                )
                continue

            destination = (
                output_root
                / member.filename
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(member) as source:
                destination.write_bytes(
                    source.read()
                )

            total_bytes += member.file_size

            extracted_files.append(
                str(
                    destination.relative_to(
                        args.source_run
                    )
                )
            )

    manifest = {
        "archive": str(
            archive_path.relative_to(
                args.source_run
            )
        ),
        "output_root": str(
            output_root.relative_to(
                args.source_run
            )
        ),
        "extracted_count": len(
            extracted_files
        ),
        "skipped_count": len(
            skipped_files
        ),
        "total_extracted_bytes": total_bytes,
        "extracted_files": extracted_files,
        "skipped_files": skipped_files,
    }

    manifest_path = (
        args.source_run
        / "inspection"
        / "6g_extraction_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "extracted_count": (
                    manifest["extracted_count"]
                ),
                "skipped_count": (
                    manifest["skipped_count"]
                ),
                "total_extracted_bytes": (
                    manifest[
                        "total_extracted_bytes"
                    ]
                ),
                "output_root": (
                    manifest["output_root"]
                ),
            },
            indent=2,
        )
    )

    print(
        f"\nSaved manifest: {manifest_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
