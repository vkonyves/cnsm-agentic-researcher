from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol


COMPLETED_STATUS = "COMPLETED"

REQUIRED_COMPLETED_ANALYSIS_FIELDS = (
    "status",
    "schema_version",
    "analysis_executor",
    "study_id",
    "input_results_sha256",
    "confirmatory_results",
    "secondary_results",
    "missingness_summary",
    "exclusions",
    "deviations_from_preregistration",
    "results_path",
    "artifact_hashes",
    "warnings",
)


class AnalysisExecutor(Protocol):
    family: str
    aliases: tuple[str, ...]

    def supports(
        self,
        *,
        analysis_plan: dict[str, Any],
        execution_manifest: dict[str, Any],
    ) -> bool:
        ...

    def execute(
        self,
        *,
        analysis_plan: dict[str, Any],
        preregistration: dict[str, Any],
        execution_manifest: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        ...


_ANALYSIS_EXECUTORS: list[
    AnalysisExecutor
] = []


def normalise_analysis_family(
    value: Any,
) -> str:
    text = str(
        value or ""
    ).strip().lower()

    characters: list[str] = []
    previous_was_separator = False

    for character in text:
        if character.isalnum():
            characters.append(
                character
            )
            previous_was_separator = False
        else:
            if (
                characters
                and not previous_was_separator
            ):
                characters.append(
                    "_"
                )
                previous_was_separator = True

    return "".join(
        characters
    ).strip("_")


def analysis_family_matches(
    value: Any,
    *,
    family: str,
    aliases: tuple[str, ...] = (),
) -> bool:
    requested = normalise_analysis_family(
        value
    )

    if not requested:
        return False

    supported = {
        normalise_analysis_family(
            identifier
        )
        for identifier in (
            family,
            *aliases,
        )
        if normalise_analysis_family(
            identifier
        )
    }

    return requested in supported


def _registered_identifiers(
    executor: AnalysisExecutor,
) -> set[str]:
    return {
        normalise_analysis_family(
            identifier
        )
        for identifier in (
            executor.family,
            *getattr(
                executor,
                "aliases",
                (),
            ),
        )
        if normalise_analysis_family(
            identifier
        )
    }


def register_analysis_executor(
    executor: AnalysisExecutor,
) -> None:
    new_identifiers = (
        _registered_identifiers(
            executor
        )
    )

    if not new_identifiers:
        raise ValueError(
            "Analysis executor must expose a "
            "non-empty family identifier."
        )

    for registered in _ANALYSIS_EXECUTORS:
        overlap = (
            new_identifiers
            & _registered_identifiers(
                registered
            )
        )

        if overlap:
            raise ValueError(
                "Analysis executor identifier "
                "already registered: "
                + ", ".join(
                    sorted(
                        overlap
                    )
                )
            )

    _ANALYSIS_EXECUTORS.append(
        executor
    )


def clear_registered_analysis_executors() -> None:
    """
    Clear the executor registry.

    Intended for isolated tests and controlled development setup.
    """
    _ANALYSIS_EXECUTORS.clear()


def registered_analysis_families() -> list[str]:
    return sorted(
        executor.family
        for executor
        in _ANALYSIS_EXECUTORS
    )


def resolve_analysis_executor(
    *,
    analysis_plan: dict[str, Any],
    execution_manifest: dict[str, Any],
) -> AnalysisExecutor | None:
    return next(
        (
            executor
            for executor
            in _ANALYSIS_EXECUTORS
            if executor.supports(
                analysis_plan=analysis_plan,
                execution_manifest=(
                    execution_manifest
                ),
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


def _safe_artifact_path(
    *,
    run_dir: Path,
    relative_path: str,
) -> Path | None:
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

    resolved = (
        run_dir
        / candidate
    ).resolve()

    run_root = run_dir.resolve()

    try:
        resolved.relative_to(
            run_root
        )
    except ValueError:
        return None

    return resolved


def validate_analysis_results(
    results: dict[str, Any],
    *,
    run_dir: Path,
    execution_manifest: dict[str, Any],
) -> list[str]:
    """
    Validate that analysis results are complete, inspectable,
    and tied to the actual execution artifacts.
    """
    issues: list[str] = []

    if not isinstance(
        results,
        dict,
    ):
        return [
            "Analysis results are not a dictionary."
        ]

    status = results.get(
        "status"
    )

    if status != COMPLETED_STATUS:
        issues.append(
            "Analysis status is not COMPLETED."
        )

    if status == COMPLETED_STATUS:
        for field in (
            REQUIRED_COMPLETED_ANALYSIS_FIELDS
        ):
            if field not in results:
                issues.append(
                    "Completed analysis results "
                    f"lack required field: {field}"
                )

    for field in (
        "confirmatory_results",
        "secondary_results",
        "exclusions",
        "deviations_from_preregistration",
        "warnings",
    ):
        value = results.get(
            field
        )

        if (
            value is not None
            and not isinstance(
                value,
                list,
            )
        ):
            issues.append(
                f"{field} must be a list."
            )

    missingness_summary = results.get(
        "missingness_summary"
    )

    if (
        missingness_summary is not None
        and not isinstance(
            missingness_summary,
            dict,
        )
    ):
        issues.append(
            "missingness_summary must be "
            "a dictionary."
        )

    confirmatory_results = results.get(
        "confirmatory_results"
    )

    if (
        status == COMPLETED_STATUS
        and isinstance(
            confirmatory_results,
            list,
        )
        and not confirmatory_results
    ):
        issues.append(
            "Completed analysis contains no "
            "confirmatory results."
        )

    execution_results_path = (
        execution_manifest.get(
            "results_path"
        )
    )

    execution_hashes = (
        execution_manifest.get(
            "artifact_hashes"
        )
    )

    expected_input_hash: str | None = None

    if (
        isinstance(
            execution_results_path,
            str,
        )
        and isinstance(
            execution_hashes,
            dict,
        )
    ):
        expected = execution_hashes.get(
            execution_results_path
        )

        if isinstance(
            expected,
            str,
        ):
            expected_input_hash = (
                expected.strip().lower()
            )

    supplied_input_hash = results.get(
        "input_results_sha256"
    )

    if (
        expected_input_hash
        and isinstance(
            supplied_input_hash,
            str,
        )
        and supplied_input_hash.strip().lower()
        != expected_input_hash
    ):
        issues.append(
            "Analysis input hash does not match "
            "the execution results hash."
        )

    if (
        expected_input_hash
        and not isinstance(
            supplied_input_hash,
            str,
        )
    ):
        issues.append(
            "Analysis input-results hash is missing."
        )

    results_path_value = results.get(
        "results_path"
    )

    resolved_results_path: Path | None = None

    if (
        not isinstance(
            results_path_value,
            str,
        )
        or not results_path_value.strip()
    ):
        issues.append(
            "results_path must be a "
            "non-empty relative path."
        )
    else:
        resolved_results_path = (
            _safe_artifact_path(
                run_dir=run_dir,
                relative_path=(
                    results_path_value
                ),
            )
        )

        if resolved_results_path is None:
            issues.append(
                "Analysis results path is unsafe: "
                f"{results_path_value}"
            )
        elif not resolved_results_path.is_file():
            issues.append(
                "Analysis results artifact does "
                f"not exist: {results_path_value}"
            )

    artifact_hashes = results.get(
        "artifact_hashes"
    )

    if (
        not isinstance(
            artifact_hashes,
            dict,
        )
        or not artifact_hashes
    ):
        issues.append(
            "Analysis artifact hashes are missing."
        )
    elif (
        resolved_results_path is not None
        and resolved_results_path.is_file()
        and isinstance(
            results_path_value,
            str,
        )
    ):
        expected_hash = artifact_hashes.get(
            results_path_value
        )

        if (
            not isinstance(
                expected_hash,
                str,
            )
            or not expected_hash.strip()
        ):
            issues.append(
                "Analysis artifact hash is "
                f"missing for: {results_path_value}"
            )
        else:
            actual_hash = _sha256_file(
                resolved_results_path
            )

            if (
                actual_hash.lower()
                != expected_hash.strip().lower()
            ):
                issues.append(
                    "Analysis artifact hash does "
                    f"not match: {results_path_value}"
                )

    return sorted(
        set(
            issues
        )
    )
