from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


AcquisitionAction = Literal[
    "download",
    "skip",
    "manual_verification_required",
]

DownloadStatus = Literal[
    "downloaded",
    "skipped",
    "failed",
]


class PlannedAsset(BaseModel):
    resource_name: str
    source_url: str
    action: AcquisitionAction
    reason: str
    expected_filename: str | None = None


class AcquisitionPlan(BaseModel):
    source_run: str
    discovery_report_path: str
    assets: list[PlannedAsset]

    downloadable_count: int
    skipped_count: int
    manual_verification_count: int


class AcquiredAsset(BaseModel):
    resource_name: str
    source_url: str
    final_url: str | None = None

    status: DownloadStatus
    reason: str | None = None

    local_relative_path: str | None = None
    filename: str | None = None

    http_status: int | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None


class AcquisitionManifest(BaseModel):
    source_run: str
    acquisition_plan_path: str
    assets: list[AcquiredAsset]

    downloaded_count: int
    skipped_count: int
    failed_count: int
