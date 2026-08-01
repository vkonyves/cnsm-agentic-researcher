from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol


COMPLETED_STATUS = "COMPLETED"


SYNTHETIC_PAIRED_ADAPTER_FAMILY = (
    "synthetic_paired_llm_benchmark_v1"
)
PAIRED_BINARY_RESULT_SCHEMA_ID = "paired_binary_episode_v1"
PAIRED_BINARY_RESULT_SCHEMA_VERSION = "1.0"
SUPPORTED_PAIRED_CONDITIONS = ("baseline", "guarded")
SUPPORTED_EXECUTION_MODES = (
    "development_rehearsal",
    "scientific_pilot",
    "final_autonomous_run",
)
FINAL_AUTONOMY_REQUIRED_FIELDS = (
    "master_prompt_sha256",
    "framework_commit",
    "framework_tag",
    "capability_manifest_sha256",
    "preregistration_sha256",
    "human_scientific_intervention_after_launch",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PAIRED_BINARY_REQUIRED_ROW_FIELDS = (
    "schema_version",
    "result_schema_id",
    "study_id",
    "episode_id",
    "pair_id",
    "task_id",
    "task_family",
    "condition",
    "paired_condition",
    "condition_order",
    "execution_mode",
    "model_provider",
    "model_name",
    "model_version",
    "model_configuration_sha256",
    "task_manifest_id",
    "task_manifest_sha256",
    "task_input_sha256",
    "reference_answer_sha256",
    "transformation_id",
    "transformation_manifest_sha256",
    "prompt_sha256",
    "call_status",
    "attempt_count",
    "model_calls_used",
    "terminal_error_type",
    "terminal_error_message",
    "response_sha256",
    "response_artifact_path",
    "scoring_status",
    "score",
    "score_reason_code",
    "scorer_id",
    "scorer_version",
    "scoring_input_sha256",
    "scoring_artifact_sha256",
    "contamination_flags",
    "validity_flags",
    "started_at_utc",
    "completed_at_utc",
    "latency_ms",
)

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

def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(
        SHA256_PATTERN.fullmatch(value.strip().lower())
    )


def validate_final_autonomy_contract(
    plan: dict[str, Any],
) -> list[str]:
    """Validate conference-compliant final-run launch constraints."""
    issues: list[str] = []

    if plan.get("execution_mode") != "final_autonomous_run":
        return issues

    for field in FINAL_AUTONOMY_REQUIRED_FIELDS:
        if field not in plan:
            issues.append(
                f"Final autonomous run lacks required field: {field}"
            )

    for field in (
        "master_prompt_sha256",
        "capability_manifest_sha256",
        "preregistration_sha256",
    ):
        value = plan.get(field)
        if value is not None and not _is_sha256(value):
            issues.append(f"{field} must be a lowercase SHA-256 digest.")

    for field in ("framework_commit", "framework_tag"):
        value = plan.get(field)
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            issues.append(f"{field} must be a non-empty string.")

    if plan.get("human_scientific_intervention_after_launch") is not False:
        issues.append(
            "Final autonomous run must prohibit human scientific "
            "intervention after launch."
        )

    if plan.get("human_text_editing_after_launch") not in (None, False):
        issues.append(
            "Final autonomous run must prohibit human text editing "
            "after launch."
        )

    return sorted(set(issues))


def validate_paired_binary_result_row(
    row: dict[str, Any],
) -> list[str]:
    """Validate one terminal condition-level paired-benchmark row."""
    issues: list[str] = []
    if not isinstance(row, dict):
        return ["Paired result row is not a dictionary."]

    for field in PAIRED_BINARY_REQUIRED_ROW_FIELDS:
        if field not in row:
            issues.append(f"Paired result row lacks required field: {field}")

    if row.get("schema_version") != PAIRED_BINARY_RESULT_SCHEMA_VERSION:
        issues.append("Unsupported paired result schema version.")
    if row.get("result_schema_id") != PAIRED_BINARY_RESULT_SCHEMA_ID:
        issues.append("Unsupported paired result schema identifier.")

    condition = row.get("condition")
    paired = row.get("paired_condition")
    if condition not in SUPPORTED_PAIRED_CONDITIONS:
        issues.append("Unsupported experimental condition.")
    if paired not in SUPPORTED_PAIRED_CONDITIONS or paired == condition:
        issues.append("paired_condition must identify the opposite condition.")
    if row.get("condition_order") not in (1, 2):
        issues.append("condition_order must be 1 or 2.")
    if row.get("execution_mode") not in SUPPORTED_EXECUTION_MODES:
        issues.append("Unsupported execution mode.")

    for field in ("attempt_count", "model_calls_used", "latency_ms"):
        value = row.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            issues.append(f"{field} must be a nonnegative integer.")
    if row.get("attempt_count") == 0:
        issues.append("attempt_count must be positive.")

    call_status = row.get("call_status")
    scoring_status = row.get("scoring_status")
    score = row.get("score")
    if call_status not in {"COMPLETED", "FAILED", "CACHED"}:
        issues.append("Unsupported call_status.")
    if score not in (0, 1, None) or isinstance(score, bool):
        issues.append("score must be 0, 1, or null.")

    if call_status in {"COMPLETED", "CACHED"}:
        if scoring_status != "COMPLETED" or score not in (0, 1):
            issues.append(
                "Successful calls require a completed binary score."
            )
        if not _is_sha256(row.get("response_sha256")):
            issues.append("Successful calls require response_sha256.")
        if not isinstance(row.get("response_artifact_path"), str):
            issues.append("Successful calls require a response artifact path.")
        if row.get("terminal_error_type") is not None:
            issues.append("Successful calls cannot have terminal errors.")
    elif call_status == "FAILED":
        if scoring_status != "NOT_SCORED" or score is not None:
            issues.append("Failed calls must remain unscored with null score.")
        if row.get("response_sha256") is not None:
            issues.append("Failed calls cannot claim a response hash.")
        if row.get("response_artifact_path") is not None:
            issues.append("Failed calls cannot claim a response artifact.")
        if not isinstance(row.get("terminal_error_type"), str):
            issues.append("Failed calls require a terminal error type.")

    for field in (
        "model_configuration_sha256",
        "task_manifest_sha256",
        "task_input_sha256",
        "reference_answer_sha256",
        "transformation_manifest_sha256",
        "prompt_sha256",
    ):
        if not _is_sha256(row.get(field)):
            issues.append(f"{field} must be a lowercase SHA-256 digest.")

    for field in ("contamination_flags", "validity_flags"):
        if not isinstance(row.get(field), list):
            issues.append(f"{field} must be a list.")

    return sorted(set(issues))


def synthetic_paired_plan_issues(
    plan: dict[str, Any],
    *,
    maximum_model_calls: int,
    supported_task_families: tuple[str, ...],
    supported_transformations: tuple[str, ...],
    available_models: tuple[tuple[str, str, str], ...],
) -> list[str]:
    """Return explicit incompatibilities for the first concrete adapter."""
    issues: list[str] = []
    if not adapter_family_matches(
        plan, family=SYNTHETIC_PAIRED_ADAPTER_FAMILY,
        aliases=("synthetic-paired-llm-benchmark-v1",),
    ):
        issues.append("Adapter family is incompatible.")

    if plan.get("result_schema_id") != PAIRED_BINARY_RESULT_SCHEMA_ID:
        issues.append("Plan does not request paired_binary_episode_v1.")
    if plan.get("result_schema_version", "1.0") != "1.0":
        issues.append("Plan requests an unsupported result schema version.")
    if tuple(plan.get("conditions", ())) != SUPPORTED_PAIRED_CONDITIONS:
        issues.append("Plan must use exactly baseline and guarded conditions.")
    if plan.get("design") != "paired_binary":
        issues.append("Plan must declare a paired_binary design.")

    estimated = plan.get("estimated_model_calls")
    if (
        not isinstance(estimated, int)
        or isinstance(estimated, bool)
        or estimated <= 0
        or estimated > maximum_model_calls
    ):
        issues.append("Estimated model calls are outside adapter scope.")

    tasks = plan.get("task_families")
    if not isinstance(tasks, list) or not tasks or any(
        task not in supported_task_families for task in tasks
    ):
        issues.append("Plan requests unsupported task families.")

    transformations = plan.get("transformations")
    if not isinstance(transformations, dict) or set(transformations) != set(
        SUPPORTED_PAIRED_CONDITIONS
    ) or any(
        value not in supported_transformations
        for value in transformations.values()
    ):
        issues.append("Plan requests unsupported condition transformations.")

    model = (
        plan.get("model_provider"),
        plan.get("model_name"),
        plan.get("model_version"),
    )
    if model not in available_models:
        issues.append("Requested provider/model/version is unavailable.")

    if plan.get("deterministic_automated_scoring") is not True:
        issues.append("Plan must require deterministic automated scoring.")
    if plan.get("requires_human_scientific_labour") is not False:
        issues.append("Plan must not require human scientific labour.")
    if plan.get("execution_mode") not in SUPPORTED_EXECUTION_MODES:
        issues.append("Plan requests an unsupported execution mode.")

    issues.extend(validate_final_autonomy_contract(plan))
    return sorted(set(issues))
