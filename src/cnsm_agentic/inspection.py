from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from cnsm_agentic.acquisition import sha256_file
from cnsm_agentic.acquisition_schemas import (
    AcquisitionManifest,
)
from cnsm_agentic.inspection_schemas import (
    InspectedAsset,
    InspectionInventory,
    JsonStructure,
    ZipMemberRecord,
    ZipStructure,
)


MAX_JSON_INSPECTION_BYTES = (
    100 * 1024 * 1024
)

MAX_ZIP_MEMBERS = 50_000

SUSPICIOUS_COMPRESSION_RATIO = 200.0


def python_type_name(
    value: Any,
) -> str:
    return type(value).__name__


def safe_archive_path(
    member_name: str,
) -> bool:
    """
    Return False for absolute paths or archive members
    attempting directory traversal.
    """

    member_path = Path(member_name)

    if member_path.is_absolute():
        return False

    if ".." in member_path.parts:
        return False

    return True


def compression_ratio(
    compressed_size: int,
    uncompressed_size: int,
) -> float | None:
    if compressed_size == 0:
        if uncompressed_size == 0:
            return None

        return float("inf")

    return (
        uncompressed_size
        / compressed_size
    )


def inspect_json_file(
    path: Path,
) -> JsonStructure:
    """
    Inspect the basic structure of a JSON file.

    This intentionally avoids inferring scientific meaning.
    """

    size_bytes = path.stat().st_size

    if size_bytes > MAX_JSON_INSPECTION_BYTES:
        raise ValueError(
            "JSON file exceeds the configured "
            "100 MiB inspection limit."
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(payload, list):
        sample = (
            payload[0]
            if payload
            else None
        )

        sample_keys: list[str] = []
        key_types: dict[str, str] = {}

        if isinstance(sample, dict):
            sample_keys = sorted(
                str(key)
                for key in sample.keys()
            )

            key_types = {
                str(key): python_type_name(value)
                for key, value in sample.items()
            }

        return JsonStructure(
            top_level_type="list",
            record_count=len(payload),
            sample_record_type=(
                python_type_name(sample)
                if sample is not None
                else None
            ),
            sample_record_keys=sample_keys,
            key_types=key_types,
        )

    if isinstance(payload, dict):
        top_level_keys = sorted(
            str(key)
            for key in payload.keys()
        )

        sample_value = next(
            iter(payload.values()),
            None,
        )

        sample_keys: list[str] = []
        key_types: dict[str, str] = {}

        if isinstance(sample_value, dict):
            sample_keys = sorted(
                str(key)
                for key in sample_value.keys()
            )

            key_types = {
                str(key): python_type_name(value)
                for key, value
                in sample_value.items()
            }

        return JsonStructure(
            top_level_type="dict",
            record_count=len(payload),
            top_level_keys=top_level_keys,
            sample_record_type=(
                python_type_name(sample_value)
                if sample_value is not None
                else None
            ),
            sample_record_keys=sample_keys,
            key_types=key_types,
        )

    return JsonStructure(
        top_level_type=python_type_name(
            payload
        ),
    )


def inspect_zip_file(
    path: Path,
) -> ZipStructure:
    """
    Inspect ZIP metadata without extracting its contents.
    """

    member_records: list[
        ZipMemberRecord
    ] = []

    suffix_counter: Counter[str] = Counter()

    compressed_total = 0
    uncompressed_total = 0

    unsafe_count = 0
    suspicious_count = 0

    with zipfile.ZipFile(
        path,
        mode="r",
    ) as archive:
        members = archive.infolist()

        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError(
                "ZIP member count exceeds "
                f"{MAX_ZIP_MEMBERS}."
            )

        for member in members:
            member_path = Path(
                member.filename
            )

            suffix = (
                member_path.suffix.lower()
                if not member.is_dir()
                else ""
            )

            suffix_counter[
                suffix or "<none>"
            ] += 1

            compressed_total += (
                member.compress_size
            )

            uncompressed_total += (
                member.file_size
            )

            safe_path = safe_archive_path(
                member.filename
            )

            ratio = compression_ratio(
                compressed_size=member.compress_size,
                uncompressed_size=member.file_size,
            )

            suspicious = (
                ratio is not None
                and ratio
                > SUSPICIOUS_COMPRESSION_RATIO
                and member.file_size
                > 1024 * 1024
            )

            if not safe_path:
                unsafe_count += 1

            if suspicious:
                suspicious_count += 1

            member_records.append(
                ZipMemberRecord(
                    name=member.filename,
                    suffix=suffix,
                    compressed_size=(
                        member.compress_size
                    ),
                    uncompressed_size=(
                        member.file_size
                    ),
                    compression_ratio=ratio,
                    is_directory=(
                        member.is_dir()
                    ),
                    safe_path=safe_path,
                    suspicious_compression=(
                        suspicious
                    ),
                )
            )

    return ZipStructure(
        member_count=len(member_records),
        compressed_size_total=(
            compressed_total
        ),
        uncompressed_size_total=(
            uncompressed_total
        ),
        unsafe_member_count=(
            unsafe_count
        ),
        suspicious_member_count=(
            suspicious_count
        ),
        suffix_counts=dict(
            sorted(
                suffix_counter.items()
            )
        ),
        members=member_records,
    )


def inspect_acquired_asset(
    source_run: Path,
    resource_name: str,
    local_relative_path: str,
    filename: str,
    expected_sha256: str,
) -> InspectedAsset:
    path = (
        source_run
        / local_relative_path
    )

    if not path.exists():
        return InspectedAsset(
            resource_name=resource_name,
            local_relative_path=(
                local_relative_path
            ),
            filename=filename,
            size_bytes=0,
            sha256="",
            suffix=path.suffix.lower(),
            status="failed",
            error=(
                f"Acquired file not found: {path}"
            ),
        )

    actual_sha256 = sha256_file(path)

    if actual_sha256 != expected_sha256:
        return InspectedAsset(
            resource_name=resource_name,
            local_relative_path=(
                local_relative_path
            ),
            filename=filename,
            size_bytes=path.stat().st_size,
            sha256=actual_sha256,
            suffix=path.suffix.lower(),
            status="failed",
            error=(
                "SHA-256 does not match the "
                "acquisition manifest."
            ),
        )

    suffix = path.suffix.lower()

    try:
        if suffix == ".json":
            return InspectedAsset(
                resource_name=resource_name,
                local_relative_path=(
                    local_relative_path
                ),
                filename=filename,
                size_bytes=path.stat().st_size,
                sha256=actual_sha256,
                suffix=suffix,
                status="inspected",
                json_structure=(
                    inspect_json_file(path)
                ),
            )

        if suffix == ".zip":
            return InspectedAsset(
                resource_name=resource_name,
                local_relative_path=(
                    local_relative_path
                ),
                filename=filename,
                size_bytes=path.stat().st_size,
                sha256=actual_sha256,
                suffix=suffix,
                status="inspected",
                zip_structure=(
                    inspect_zip_file(path)
                ),
            )

        return InspectedAsset(
            resource_name=resource_name,
            local_relative_path=(
                local_relative_path
            ),
            filename=filename,
            size_bytes=path.stat().st_size,
            sha256=actual_sha256,
            suffix=suffix,
            status="unsupported",
            error=(
                "No structural inspector is "
                f"implemented for {suffix!r}."
            ),
        )

    except Exception as exc:
        return InspectedAsset(
            resource_name=resource_name,
            local_relative_path=(
                local_relative_path
            ),
            filename=filename,
            size_bytes=path.stat().st_size,
            sha256=actual_sha256,
            suffix=suffix,
            status="failed",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


def build_inspection_inventory(
    source_run: Path,
) -> InspectionInventory:
    manifest_path = (
        source_run
        / "acquisition"
        / "acquisition_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Acquisition manifest not found: "
            f"{manifest_path}"
        )

    manifest = (
        AcquisitionManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    inspected_assets: list[
        InspectedAsset
    ] = []

    for asset in manifest.assets:
        if asset.status != "downloaded":
            continue

        if not asset.local_relative_path:
            continue

        if not asset.filename:
            continue

        if not asset.sha256:
            continue

        inspected_assets.append(
            inspect_acquired_asset(
                source_run=source_run,
                resource_name=(
                    asset.resource_name
                ),
                local_relative_path=(
                    asset.local_relative_path
                ),
                filename=asset.filename,
                expected_sha256=asset.sha256,
            )
        )

    inventory = InspectionInventory(
        source_run=str(source_run),
        acquisition_manifest_path=str(
            manifest_path
        ),
        assets=inspected_assets,
        inspected_count=sum(
            asset.status == "inspected"
            for asset in inspected_assets
        ),
        unsupported_count=sum(
            asset.status == "unsupported"
            for asset in inspected_assets
        ),
        failed_count=sum(
            asset.status == "failed"
            for asset in inspected_assets
        ),
        unsafe_zip_member_count=sum(
            (
                asset.zip_structure
                .unsafe_member_count
            )
            if asset.zip_structure
            else 0
            for asset in inspected_assets
        ),
        suspicious_zip_member_count=sum(
            (
                asset.zip_structure
                .suspicious_member_count
            )
            if asset.zip_structure
            else 0
            for asset in inspected_assets
        ),
    )

    inspection_directory = (
        source_run
        / "inspection"
    )

    inspection_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        inspection_directory
        / "inspection_inventory.json"
    )

    output_path.write_text(
        inventory.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return inventory
