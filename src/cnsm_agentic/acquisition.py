from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from cnsm_agentic.acquisition_schemas import (
    AcquiredAsset,
    AcquisitionManifest,
    AcquisitionPlan,
    PlannedAsset,
)
from cnsm_agentic.schemas import DiscoveryReport


DIRECT_DOWNLOAD_SUFFIXES = {
    ".zip",
    ".json",
    ".jsonl",
    ".csv",
    ".txt",
    ".yaml",
    ".yml",
    ".parquet",
    ".gz",
    ".tar",
}

ALLOWED_HOSTS = {
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
}

CHUNK_SIZE = 1024 * 1024

# Safety cap for a single downloaded file.
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def probable_filename(url: str) -> str | None:
    """Extract a probable filename from a URL path."""

    path = unquote(urlparse(url).path)
    filename = Path(path).name

    if not filename:
        return None

    return filename


def is_direct_download_url(url: str) -> bool:
    """Return True when a URL path looks like a direct file download."""

    path = unquote(urlparse(url).path).lower()

    return any(
        path.endswith(suffix)
        for suffix in DIRECT_DOWNLOAD_SUFFIXES
    )


def classify_asset(
    resource_name: str,
    source_url: str,
) -> PlannedAsset:
    """Classify a discovered dataset URL for deterministic acquisition."""

    parsed = urlparse(source_url)

    if parsed.scheme not in {"http", "https"}:
        return PlannedAsset(
            resource_name=resource_name,
            source_url=source_url,
            action="skip",
            reason="URL does not use HTTP or HTTPS.",
        )

    if not parsed.netloc:
        return PlannedAsset(
            resource_name=resource_name,
            source_url=source_url,
            action="skip",
            reason="URL does not contain a hostname.",
        )

    if is_direct_download_url(source_url):
        return PlannedAsset(
            resource_name=resource_name,
            source_url=source_url,
            action="download",
            reason=(
                "URL appears to reference a direct downloadable file."
            ),
            expected_filename=probable_filename(source_url),
        )

    return PlannedAsset(
        resource_name=resource_name,
        source_url=source_url,
        action="manual_verification_required",
        reason=(
            "URL appears to be a landing page, DOI, repository page, "
            "or directory rather than a direct downloadable file."
        ),
        expected_filename=probable_filename(source_url),
    )


def build_acquisition_plan(
    source_run: Path,
) -> AcquisitionPlan:
    """Build a deterministic acquisition plan from a discovery report."""

    discovery_path = (
        source_run
        / "discovery"
        / "discovery_report.json"
    )

    if not discovery_path.exists():
        raise FileNotFoundError(
            f"Discovery report not found: {discovery_path}"
        )

    discovery = DiscoveryReport.model_validate_json(
        discovery_path.read_text(
            encoding="utf-8"
        )
    )

    assets: list[PlannedAsset] = []

    for resource in discovery.resources:
        for dataset_url in resource.dataset_urls:
            assets.append(
                classify_asset(
                    resource_name=resource.name,
                    source_url=dataset_url,
                )
            )

    return AcquisitionPlan(
        source_run=str(source_run),
        discovery_report_path=str(discovery_path),
        assets=assets,
        downloadable_count=sum(
            asset.action == "download"
            for asset in assets
        ),
        skipped_count=sum(
            asset.action == "skip"
            for asset in assets
        ),
        manual_verification_count=sum(
            asset.action == "manual_verification_required"
            for asset in assets
        ),
    )


def safe_filename(value: str) -> str:
    """Convert an arbitrary resource or file name into a safe local name."""

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        value,
    ).strip("-")

    return cleaned or "download.bin"


def validate_download_url(
    url: str,
) -> tuple[bool, str | None]:
    """Validate a URL before allowing a network download."""

    parsed = urlparse(url)

    if parsed.scheme != "https":
        return False, "Only HTTPS downloads are allowed."

    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return False, "Download URL does not contain a hostname."

    if hostname not in ALLOWED_HOSTS:
        return (
            False,
            f"Host is not allow-listed: {hostname}",
        )

    if not is_direct_download_url(url):
        return (
            False,
            "URL does not appear to reference a direct file.",
        )

    return True, None


def response_filename(
    source_url: str,
    response: httpx.Response,
) -> str:
    """Determine a local filename from response headers or URL paths."""

    content_disposition = response.headers.get(
        "content-disposition",
        "",
    )

    filename_match = re.search(
        r'filename="?([^";]+)"?',
        content_disposition,
        flags=re.IGNORECASE,
    )

    if filename_match:
        return safe_filename(
            unquote(
                filename_match.group(1)
            )
        )

    final_path = unquote(
        urlparse(str(response.url)).path
    )

    final_name = Path(final_path).name

    if final_name:
        return safe_filename(final_name)

    probable = probable_filename(source_url)

    return safe_filename(
        probable or "download.bin"
    )


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a local file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def append_acquisition_event(
    event_path: Path,
    payload: dict[str, object],
) -> None:
    """Append one acquisition event as JSONL."""

    event_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with event_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )


async def download_asset(
    client: httpx.AsyncClient,
    planned_asset: PlannedAsset,
    source_run: Path,
    download_root: Path,
    event_path: Path,
) -> AcquiredAsset:
    """Download one planned asset and record its provenance."""

    if planned_asset.action != "download":
        return AcquiredAsset(
            resource_name=planned_asset.resource_name,
            source_url=planned_asset.source_url,
            status="skipped",
            reason=planned_asset.reason,
        )

    allowed, rejection_reason = validate_download_url(
        planned_asset.source_url
    )

    if not allowed:
        return AcquiredAsset(
            resource_name=planned_asset.resource_name,
            source_url=planned_asset.source_url,
            status="skipped",
            reason=rejection_reason,
        )

    append_acquisition_event(
        event_path,
        {
            "type": "download_started",
            "resource_name": planned_asset.resource_name,
            "source_url": planned_asset.source_url,
        },
    )

    temporary_path: Path | None = None

    try:
        async with client.stream(
            "GET",
            planned_asset.source_url,
        ) as response:
            response.raise_for_status()

            declared_size = response.headers.get(
                "content-length"
            )

            if declared_size is not None:
                declared_bytes = int(declared_size)

                if declared_bytes > MAX_DOWNLOAD_BYTES:
                    skipped = AcquiredAsset(
                        resource_name=planned_asset.resource_name,
                        source_url=planned_asset.source_url,
                        final_url=str(response.url),
                        status="skipped",
                        reason=(
                            "Declared download size exceeds "
                            f"{MAX_DOWNLOAD_BYTES} bytes."
                        ),
                        http_status=response.status_code,
                    )

                    append_acquisition_event(
                        event_path,
                        {
                            "type": "download_skipped",
                            **skipped.model_dump(
                                mode="json"
                            ),
                        },
                    )

                    return skipped

            resource_directory = (
                download_root
                / safe_filename(
                    planned_asset.resource_name
                )
            )

            resource_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = response_filename(
                planned_asset.source_url,
                response,
            )

            destination = (
                resource_directory
                / filename
            )

            temporary_path = destination.with_suffix(
                destination.suffix + ".part"
            )

            received_bytes = 0

            with temporary_path.open("wb") as handle:
                async for chunk in response.aiter_bytes(
                    CHUNK_SIZE
                ):
                    received_bytes += len(chunk)

                    if received_bytes > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(
                            "Download exceeded the configured "
                            "maximum file size."
                        )

                    handle.write(chunk)

            temporary_path.replace(destination)

            media_type = response.headers.get(
                "content-type"
            )

            if not media_type:
                media_type, _ = mimetypes.guess_type(
                    destination.name
                )

            acquired = AcquiredAsset(
                resource_name=planned_asset.resource_name,
                source_url=planned_asset.source_url,
                final_url=str(response.url),
                status="downloaded",
                local_relative_path=str(
                    destination.relative_to(
                        source_run
                    )
                ),
                filename=destination.name,
                http_status=response.status_code,
                media_type=media_type,
                size_bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
            )

            append_acquisition_event(
                event_path,
                {
                    "type": "download_completed",
                    **acquired.model_dump(
                        mode="json"
                    ),
                },
            )

            return acquired

    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )

        failed = AcquiredAsset(
            resource_name=planned_asset.resource_name,
            source_url=planned_asset.source_url,
            status="failed",
            reason=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        append_acquisition_event(
            event_path,
            {
                "type": "download_failed",
                **failed.model_dump(
                    mode="json"
                ),
            },
        )

        return failed


async def execute_acquisition_plan(
    source_run: Path,
) -> AcquisitionManifest:
    """Execute a previously generated acquisition plan."""

    plan_path = (
        source_run
        / "acquisition"
        / "acquisition_plan.json"
    )

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Acquisition plan not found: {plan_path}"
        )

    plan = AcquisitionPlan.model_validate_json(
        plan_path.read_text(
            encoding="utf-8"
        )
    )

    acquisition_root = (
        source_run
        / "acquisition"
    )

    download_root = (
        acquisition_root
        / "files"
    )

    event_path = (
        acquisition_root
        / "acquisition_events.jsonl"
    )

    download_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeout = httpx.Timeout(
        connect=20.0,
        read=300.0,
        write=60.0,
        pool=20.0,
    )

    headers = {
        "User-Agent": (
            "cnsm-agentic-researcher/0.3"
        ),
        "Accept": "*/*",
    }

    results: list[AcquiredAsset] = []

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:
        for planned_asset in plan.assets:
            result = await download_asset(
                client=client,
                planned_asset=planned_asset,
                source_run=source_run,
                download_root=download_root,
                event_path=event_path,
            )

            results.append(result)

    manifest = AcquisitionManifest(
        source_run=str(source_run),
        acquisition_plan_path=str(plan_path),
        assets=results,
        downloaded_count=sum(
            asset.status == "downloaded"
            for asset in results
        ),
        skipped_count=sum(
            asset.status == "skipped"
            for asset in results
        ),
        failed_count=sum(
            asset.status == "failed"
            for asset in results
        ),
    )

    manifest_path = (
        acquisition_root
        / "acquisition_manifest.json"
    )

    manifest_path.write_text(
        manifest.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return manifest
