from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


InspectionStatus = Literal[
    "inspected",
    "unsupported",
    "failed",
]


class JsonStructure(BaseModel):
    top_level_type: str
    record_count: int | None = None

    top_level_keys: list[str] = Field(
        default_factory=list
    )

    sample_record_type: str | None = None

    sample_record_keys: list[str] = Field(
        default_factory=list
    )

    key_types: dict[str, str] = Field(
        default_factory=dict
    )


class ZipMemberRecord(BaseModel):
    name: str
    suffix: str

    compressed_size: int
    uncompressed_size: int

    compression_ratio: float | None = None

    is_directory: bool
    safe_path: bool
    suspicious_compression: bool


class ZipStructure(BaseModel):
    member_count: int

    compressed_size_total: int
    uncompressed_size_total: int

    unsafe_member_count: int
    suspicious_member_count: int

    suffix_counts: dict[str, int] = Field(
        default_factory=dict
    )

    members: list[ZipMemberRecord] = Field(
        default_factory=list
    )


class InspectedAsset(BaseModel):
    resource_name: str
    local_relative_path: str
    filename: str

    size_bytes: int
    sha256: str
    suffix: str

    status: InspectionStatus
    error: str | None = None

    json_structure: JsonStructure | None = None
    zip_structure: ZipStructure | None = None


class InspectionInventory(BaseModel):
    source_run: str
    acquisition_manifest_path: str

    assets: list[InspectedAsset]

    inspected_count: int
    unsupported_count: int
    failed_count: int

    unsafe_zip_member_count: int
    suspicious_zip_member_count: int
