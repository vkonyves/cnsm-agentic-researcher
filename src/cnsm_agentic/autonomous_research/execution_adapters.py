from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol


COMPLETED_STATUS = "COMPLETED"

REQUIRED_COMPLETED_MANIFEST_FIELDS = (
    "schema_version",
    "adapter_family",
    "study_id",
    "started_at_utc",
    "completed_at_utc",
    "planned_episode_count",
    "completed_episode_count",
    "failed_episode_count",
    "model_calls_used",
    "results_path",
    "result_schema_path",
    "execution_log_path",
    "artifact_hashes",
    "warnings",
)


class ExecutionAdapter(Protocol):
    family: str
    aliases: tuple[str, ...]

    def supports(
        self,
        plan: dict[str, Any],
    ) -> bool:
        ...

    def execute(
        self,
        *,
        plan: dict[str, Any],
        preregistration: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        ...


_ADAPTERS: list[ExecutionAdapter] = []


def normalise_adapter_family(
    value: Any,
) -> str:
    """
    Normalize adapter identifiers for exact identifier comparison.

    This does not perform fuzzy or semantic matching.
    """
    text = str(
        value or ""
    ).strip().lower()

    normalized_characters: list[str] = []
    previous_was_separator = False

    for character in text:
        if character.isalnum():
            normalized_characters.append(
                character
            )
            previous_was_separator = False
        else:
            if (
                normalized_characters
                and not previous_was_separator
            ):
                normalized_characters.append(
                    "_"
                )
                previous_was_separator = True

    return "".join(
        normalized_characters
    ).strip("_")


def adapter_family_matches(
    plan: dict[str, Any],
    *,
    family: str,
    aliases: tuple[str, ...] = (),
) -> bool:
    """
    Match a plan only against an adapter's explicit identifier or aliases.
    """
    requested_family = (
        normalise_adapter_family(
            plan.get(
                "adapter_family"
            )
        )
    )

    if not requested_family:
        return False

    supported_families = {
        normalise_adapter_family(
            candidate
        )
        for candidate in (
            family,
            *aliases,
        )
        if normalise_adapter_family(
            candidate
        )
    }

    return (
        requested_family
        in supported_families
    )


def _registered_identifiers(
    adapter: ExecutionAdapter,
) -> set[str]:
    return {
        normalise_adapter_family(
            identifier
        )
        for identifier in (
            adapter.family,
            *getattr(
                adapter,
                "aliases",
                (),
            ),
        )
        if normalise_adapter_family(
            identifier
        )
    }


def register_adapter(
    adapter: ExecutionAdapter,
) -> None:
    """
    Register one execution adapter.

    Adapter identifiers and aliases must not collide with an existing
    registered adapter.
    """
    new_identifiers = (
        _registered_identifiers(
            adapter
        )
    )

    if not new_identifiers:
        raise ValueError(
            "Execution adapter must expose a "
            "non-empty family identifier."
        )

    for registered in _ADAPTERS:
        overlap = (
            new_identifiers
            & _registered_identifiers(
                registered
            )
        )

        if overlap:
            raise ValueError(
                "Execution adapter identifier "
                "already registered: "
                + ", ".join(
                    sorted(
                        overlap
                    )
                )
            )

    _ADAPTERS.append(
        adapter
    )


def clear_registered_adapters() -> None:
    """
    Clear the adapter registry.

    Intended for isolated tests and controlled development setup.
    """
    _ADAPTERS.clear()


def registered_adapter_families() -> list[str]:
    return sorted(
        adapter.family
        for adapter in _ADAPTERS
    )


def resolve_adapter(
    plan: dict[str, Any],
) -> ExecutionAdapter | None:
    """
    Resolve a plan through registered adapters only.

    The adapter's supports() implementation is responsible for exact,
    explicit compatibility checks. No fuzzy fallback is applied.
    """
    return next(
        (
            adapter
            for adapter in _ADAPTERS
            if adapter.supports(
                plan
            )
        ),
        None,
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _resolve_artifact_path(
    *,
    output_dir: Path,
    relative_path: str,
) -> Path | None:
    """
    Resolve a manifest path without allowing absolute paths or traversal.

    Both of these representations are accepted:

    - raw_results.jsonl
    - execution/raw_results.jsonl
    """
    candidate = Path(
        relative_path
    )

    if candidate.is_absolute():
        return None

    if any(
        part == ".."
        for part in candidate.parts
    ):
        return None

    if (
        candidate.parts
        and candidate.parts[0]
        == output_dir.name
    ):
        candidate = Path(
            *candidate.parts[1:]
        )

    resolved = (
        output_dir
        / candidate
    ).resolve()

    output_root = (
        output_dir.resolve()
    )

    try:
        resolved.relative_to(
            output_root
        )
    except ValueError:
        return None

    return resolved


def validate_execution_manifest(
    manifest: dict[str, Any],
    *,
    plan: dict[str, Any],
    output_dir: Path,
    maximum_model_calls: int | None = None,
) -> list[str]:
    """
    Validate that a completed execution represents actual, inspectable work.
    """
    issues: list[str] = []

    if not isinstance(
        manifest,
        dict,
    ):
        return [
            "Execution manifest is not a dictionary."
        ]

    status = manifest.get(
        "status"
    )

    if status != COMPLETED_STATUS:
        issues.append(
            "Execution status is not COMPLETED."
        )

    if status == COMPLETED_STATUS:
        for field in (
            REQUIRED_COMPLETED_MANIFEST_FIELDS
        ):
            if field not in manifest:
                issues.append(
                    "Completed execution manifest "
                    f"lacks required field: {field}"
                )

    manifest_family = (
        normalise_adapter_family(
            manifest.get(
                "adapter_family"
            )
        )
    )

    plan_family = (
        normalise_adapter_family(
            plan.get(
                "adapter_family"
            )
        )
    )

    if (
        manifest_family
        and plan_family
        and manifest_family
        != plan_family
    ):
        issues.append(
            "Execution manifest adapter family "
            "does not match the experiment plan."
        )

    count_fields = (
        "planned_episode_count",
        "completed_episode_count",
        "failed_episode_count",
        "model_calls_used",
    )

    counts: dict[str, int] = {}

    for field in count_fields:
        value = manifest.get(
            field
        )

        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
            or value < 0
        ):
            issues.append(
                f"{field} must be a "
                "nonnegative integer."
            )
            continue

        counts[
            field
        ] = value

    planned = counts.get(
        "planned_episode_count"
    )
    completed = counts.get(
        "completed_episode_count"
    )
    failed = counts.get(
        "failed_episode_count"
    )

    if (
        planned is not None
        and completed is not None
        and failed is not None
        and completed + failed != planned
    ):
        issues.append(
            "Completed and failed episode counts "
            "do not equal the planned episode count."
        )

    if (
        completed is not None
        and completed == 0
    ):
        issues.append(
            "Completed execution contains no "
            "successful episodes."
        )

    model_calls_used = counts.get(
        "model_calls_used"
    )

    if (
        isinstance(
            maximum_model_calls,
            int,
        )
        and model_calls_used is not None
        and model_calls_used
        > maximum_model_calls
    ):
        issues.append(
            "Execution model calls exceed the "
            "frozen capability limit: "
            f"{model_calls_used} > "
            f"{maximum_model_calls}"
        )

    artifact_fields = (
        "results_path",
        "result_schema_path",
        "execution_log_path",
    )

    resolved_artifacts: dict[
        str,
        Path,
    ] = {}

    for field in artifact_fields:
        value = manifest.get(
            field
        )

        if not isinstance(
            value,
            str,
        ) or not value.strip():
            issues.append(
                f"{field} must be a "
                "non-empty relative path."
            )
            continue

        resolved = _resolve_artifact_path(
            output_dir=output_dir,
            relative_path=value,
        )

        if resolved is None:
            issues.append(
                "Execution artifact path is "
                f"unsafe: {value}"
            )
            continue

        if not resolved.is_file():
            issues.append(
                "Execution artifact does not "
                f"exist: {value}"
            )
            continue

        resolved_artifacts[
            value
        ] = resolved

    artifact_hashes = manifest.get(
        "artifact_hashes"
    )

    if not isinstance(
        artifact_hashes,
        dict,
    ) or not artifact_hashes:
        issues.append(
            "Execution artifact hashes are missing."
        )
    else:
        for manifest_path, artifact_path in (
            resolved_artifacts.items()
        ):
            expected_hash = (
                artifact_hashes.get(
                    manifest_path
                )
            )

            if not isinstance(
                expected_hash,
                str,
            ) or not expected_hash.strip():
                issues.append(
                    "Execution artifact hash "
                    f"is missing for: {manifest_path}"
                )
                continue

            actual_hash = _sha256_file(
                artifact_path
            )

            if (
                actual_hash.lower()
                != expected_hash.strip().lower()
            ):
                issues.append(
                    "Execution artifact hash "
                    f"does not match: {manifest_path}"
                )

    warnings = manifest.get(
        "warnings"
    )

    if (
        warnings is not None
        and not isinstance(
            warnings,
            list,
        )
    ):
        issues.append(
            "Execution manifest warnings "
            "must be a list."
        )

    for timestamp_field in (
        "started_at_utc",
        "completed_at_utc",
    ):
        timestamp = manifest.get(
            timestamp_field
        )

        if (
            timestamp is not None
            and (
                not isinstance(
                    timestamp,
                    str,
                )
                or not timestamp.strip()
            )
        ):
            issues.append(
                f"{timestamp_field} must be "
                "a non-empty string."
            )

    return sorted(
        set(
            issues
        )
    )