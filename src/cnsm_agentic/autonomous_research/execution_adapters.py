from __future__ import annotations

import hashlib
import json
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
    "scoring_artifact_path",
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
        for manifest_path, expected_hash in artifact_hashes.items():
            if (
                not isinstance(manifest_path, str)
                or not manifest_path.strip()
            ):
                issues.append(
                    "Execution artifact hash contains an invalid path key."
                )
                continue

            artifact_path = _resolve_artifact_path(
                output_dir=output_dir,
                relative_path=manifest_path,
            )
            if artifact_path is None:
                issues.append(
                    "Execution artifact hash path is unsafe: "
                    f"{manifest_path}"
                )
                continue
            if not artifact_path.is_file():
                issues.append(
                    "Execution artifact hash path does not exist: "
                    f"{manifest_path}"
                )
                continue
            if (
                not isinstance(expected_hash, str)
                or not expected_hash.strip()
            ):
                issues.append(
                    "Execution artifact hash is missing for: "
                    f"{manifest_path}"
                )
                continue

            actual_hash = _sha256_file(artifact_path)
            if actual_hash.lower() != expected_hash.strip().lower():
                issues.append(
                    "Execution artifact hash does not match: "
                    f"{manifest_path}"
                )

        for manifest_path in resolved_artifacts:
            if manifest_path not in artifact_hashes:
                issues.append(
                    "Execution artifact hash is missing for: "
                    f"{manifest_path}"
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
        if not _is_sha256(row.get("response_sha256")):
            issues.append("Successful calls require response_sha256.")
        if not isinstance(row.get("response_artifact_path"), str):
            issues.append("Successful calls require a response artifact path.")
        if row.get("terminal_error_type") is not None:
            issues.append("Successful calls cannot have terminal errors.")
        if scoring_status == "COMPLETED":
            if score not in (0, 1):
                issues.append(
                    "Completed scoring requires a binary score."
                )
        elif scoring_status == "NOT_SCORED":
            if score is not None:
                issues.append(
                    "Unscored responses require a null score."
                )
        else:
            issues.append(
                "Successful calls require COMPLETED or NOT_SCORED scoring."
            )
        if call_status == "CACHED" and row.get("model_calls_used") != 0:
            issues.append("Cached calls must use zero model calls.")
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

    scoring_artifact_path = row.get("scoring_artifact_path")
    if (
        not isinstance(scoring_artifact_path, str)
        or not scoring_artifact_path.strip()
    ):
        issues.append(
            "Every terminal episode requires a scoring artifact path."
        )
    if not _is_sha256(row.get("scoring_artifact_sha256")):
        issues.append(
            "Every terminal episode requires a scoring artifact hash."
        )

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



def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_to_parent(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir.parent).as_posix()


class SyntheticPairedLLMBenchmarkAdapter:
    """Deterministic development implementation of the paired contract.

    This first implementation proves task, response, scoring, logging, hashing,
    and manifest flow. It intentionally supports development rehearsal only and
    therefore cannot be used as final scientific evidence.
    """

    family = SYNTHETIC_PAIRED_ADAPTER_FAMILY
    aliases = ("synthetic-paired-llm-benchmark-v1",)
    maximum_model_calls = 200
    supported_task_families = ("configuration_error_detection_v1",)
    supported_transformations = (
        "baseline_prompt_v1",
        "guarded_prompt_v1",
    )
    available_models = ((
        "deterministic_local",
        "paired-smoke-model",
        "1.0",
    ),)

    def supports(self, plan: dict[str, Any]) -> bool:
        issues = synthetic_paired_plan_issues(
            plan,
            maximum_model_calls=self.maximum_model_calls,
            supported_task_families=self.supported_task_families,
            supported_transformations=self.supported_transformations,
            available_models=self.available_models,
        )
        if plan.get("execution_mode") != "development_rehearsal":
            issues.append(
                "Deterministic stand-in supports development rehearsal only."
            )
        task_count = plan.get("task_count")
        if (
            not isinstance(task_count, int)
            or isinstance(task_count, bool)
            or task_count <= 0
        ):
            issues.append("task_count must be a positive integer.")
        elif plan.get("estimated_model_calls") != task_count * 2:
            issues.append(
                "estimated_model_calls must equal two calls per paired task."
            )
        return not issues

    def execute(
        self,
        *,
        plan: dict[str, Any],
        preregistration: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        if not self.supports(plan):
            raise ValueError(
                "Unsupported synthetic paired plan: "
                + "; ".join(
                    synthetic_paired_plan_issues(
                        plan,
                        maximum_model_calls=self.maximum_model_calls,
                        supported_task_families=self.supported_task_families,
                        supported_transformations=self.supported_transformations,
                        available_models=self.available_models,
                    )
                )
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = output_dir / "responses"
        responses_dir.mkdir(exist_ok=True)
        scoring_dir = output_dir / "scoring"
        scoring_dir.mkdir(exist_ok=True)

        study_id = str(plan["study_id"])
        task_count = int(plan["task_count"])
        started = str(plan.get(
            "rehearsal_started_at_utc",
            "2026-08-01T00:00:00+00:00",
        ))

        tasks: list[dict[str, Any]] = []
        for index in range(1, task_count + 1):
            task_id = f"task-{index:06d}"
            pair_id = f"pair-{index:06d}"
            payload = {
                "candidate_configuration": (
                    f"interface eth{index} mtu {1400 + index}"
                ),
                "policy": "MTU must be at least 1500.",
                "question": "Does the configuration violate the policy?",
            }
            reference_answer = "YES"
            tasks.append({
                "schema_version": "1.0",
                "task_manifest_id": "task-manifest-v1",
                "study_id": study_id,
                "task_id": task_id,
                "pair_id": pair_id,
                "task_family": "configuration_error_detection_v1",
                "generator_id": "deterministic_smoke_task_generator_v1",
                "generator_version": "1.0",
                "generation_seed": index,
                "source_identifier": f"synthetic:{task_id}",
                "task_payload": payload,
                "reference_answer": reference_answer,
                "task_input_sha256": _sha256_bytes(
                    _canonical_json_bytes(payload)
                ),
                "reference_answer_sha256": _sha256_bytes(
                    reference_answer.encode("utf-8")
                ),
                "contamination_checks": [],
            })

        task_manifest_path = output_dir / "task_manifest.jsonl"
        task_manifest_path.write_text(
            "".join(
                json.dumps(task, sort_keys=True) + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )
        task_manifest_hash = _sha256_file(task_manifest_path)

        transformation_manifest = {
            "schema_version": "1.0",
            "study_id": study_id,
            "conditions": {
                "baseline": {
                    "transformation_id": "baseline_prompt_v1",
                    "instruction": "Answer YES or NO.",
                },
                "guarded": {
                    "transformation_id": "guarded_prompt_v1",
                    "instruction": (
                        "Check the numeric MTU against the policy, then answer "
                        "YES or NO. Return only the answer."
                    ),
                },
            },
        }
        transformation_path = output_dir / "transformation_manifest.json"
        _write_json(transformation_path, transformation_manifest)
        transformation_hash = _sha256_file(transformation_path)

        model_configuration = {
            "provider": "deterministic_local",
            "model_name": "paired-smoke-model",
            "model_version": "1.0",
            "execution_mode": "development_rehearsal",
            "scientific_evidence": False,
        }
        model_configuration_path = output_dir / "model_configuration.json"
        _write_json(model_configuration_path, model_configuration)
        model_configuration_hash = _sha256_file(model_configuration_path)

        result_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": PAIRED_BINARY_RESULT_SCHEMA_ID,
            "schema_version": PAIRED_BINARY_RESULT_SCHEMA_VERSION,
            "type": "object",
            "required": list(PAIRED_BINARY_REQUIRED_ROW_FIELDS),
        }
        result_schema_path = output_dir / "result_schema.json"
        _write_json(result_schema_path, result_schema)

        failure_task_ids = set(plan.get("rehearsal_failure_task_ids", []))
        unscorable_task_ids = set(
            plan.get("rehearsal_unscorable_task_ids", [])
        )
        cached_task_ids = set(plan.get("rehearsal_cached_task_ids", []))
        rows: list[dict[str, Any]] = []
        log_events: list[dict[str, Any]] = []
        model_calls_used = 0

        for task in tasks:
            for order, condition in enumerate(SUPPORTED_PAIRED_CONDITIONS, 1):
                paired_condition = (
                    "guarded" if condition == "baseline" else "baseline"
                )
                episode_id = f"{task['task_id']}-{condition}"
                transformation_id = plan["transformations"][condition]
                instruction = transformation_manifest["conditions"][condition][
                    "instruction"
                ]
                prompt = (
                    f"{instruction}\n"
                    f"Configuration: {task['task_payload']['candidate_configuration']}\n"
                    f"Policy: {task['task_payload']['policy']}\n"
                    f"Question: {task['task_payload']['question']}"
                )
                prompt_hash = _sha256_bytes(prompt.encode("utf-8"))

                failed = (
                    task["task_id"] in failure_task_ids
                    and condition == "baseline"
                )
                unscorable = (
                    task["task_id"] in unscorable_task_ids
                    and condition == "baseline"
                )
                cached = (
                    task["task_id"] in cached_task_ids
                    and condition == "guarded"
                )

                if failed:
                    model_calls_for_episode = 1
                    response = None
                    normalized_response = None
                    score = None
                    call_status = "FAILED"
                    scoring_status = "NOT_SCORED"
                    reason = "CALL_FAILED"
                    response_hash = None
                    response_relative = None
                    scoring_input_hash = None
                    error_type = "InjectedRehearsalFailure"
                    error_message = (
                        "Deterministic failure requested by rehearsal plan."
                    )
                else:
                    model_calls_for_episode = 0 if cached else 1
                    if unscorable:
                        response = "MAYBE"
                    else:
                        # Baseline deliberately misses every third task; guarded
                        # is correct on every task, creating fixed discordance.
                        response = (
                            "NO"
                            if condition == "baseline"
                            and int(task["task_id"].split("-")[-1]) % 3 == 0
                            else "YES"
                        )
                    normalized_response = response.strip().upper()
                    call_status = "CACHED" if cached else "COMPLETED"
                    response_path = responses_dir / f"{episode_id}.txt"
                    response_path.write_text(
                        response + "\n",
                        encoding="utf-8",
                    )
                    response_hash = _sha256_file(response_path)
                    response_relative = _relative_to_parent(
                        response_path,
                        output_dir,
                    )
                    if normalized_response not in {"YES", "NO"}:
                        score = None
                        scoring_status = "NOT_SCORED"
                        reason = "RESPONSE_FORMAT_INVALID"
                        scoring_input_hash = None
                    else:
                        score = int(
                            normalized_response == task["reference_answer"]
                        )
                        scoring_status = "COMPLETED"
                        reason = (
                            "EXACT_MATCH" if score else "EXACT_MISMATCH"
                        )
                        scoring_input = {
                            "response": normalized_response,
                            "reference_answer": task["reference_answer"],
                        }
                        scoring_input_hash = _sha256_bytes(
                            _canonical_json_bytes(scoring_input)
                        )
                    error_type = None
                    error_message = None

                model_calls_used += model_calls_for_episode

                scoring_record = {
                    "schema_version": "1.0",
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": task["pair_id"],
                    "task_id": task["task_id"],
                    "condition": condition,
                    "scorer_id": "deterministic_netops_scorer_v1",
                    "scorer_version": "1.0",
                    "scoring_rule": "normalized_exact_match",
                    "raw_response_sha256": response_hash,
                    "normalized_response": normalized_response,
                    "reference_answer_sha256": task[
                        "reference_answer_sha256"
                    ],
                    "normalized_reference_answer": task[
                        "reference_answer"
                    ],
                    "score": score,
                    "score_reason_code": reason,
                    "scoring_status": scoring_status,
                    "terminal_error_type": error_type,
                }
                scoring_path = scoring_dir / f"{episode_id}.json"
                _write_json(scoring_path, scoring_record)
                scoring_relative = _relative_to_parent(
                    scoring_path, output_dir
                )
                scoring_artifact_hash = _sha256_file(scoring_path)

                row = {
                    "schema_version": "1.0",
                    "result_schema_id": PAIRED_BINARY_RESULT_SCHEMA_ID,
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": task["pair_id"],
                    "task_id": task["task_id"],
                    "task_family": task["task_family"],
                    "condition": condition,
                    "paired_condition": paired_condition,
                    "condition_order": order,
                    "execution_mode": "development_rehearsal",
                    "model_provider": "deterministic_local",
                    "model_name": "paired-smoke-model",
                    "model_version": "1.0",
                    "model_configuration_sha256": model_configuration_hash,
                    "task_manifest_id": "task-manifest-v1",
                    "task_manifest_sha256": task_manifest_hash,
                    "task_input_sha256": task["task_input_sha256"],
                    "reference_answer_sha256": task[
                        "reference_answer_sha256"
                    ],
                    "transformation_id": transformation_id,
                    "transformation_manifest_sha256": transformation_hash,
                    "prompt_sha256": prompt_hash,
                    "call_status": call_status,
                    "attempt_count": 1,
                    "model_calls_used": model_calls_for_episode,
                    "terminal_error_type": error_type,
                    "terminal_error_message": error_message,
                    "response_sha256": response_hash,
                    "response_artifact_path": response_relative,
                    "scoring_status": scoring_status,
                    "score": score,
                    "score_reason_code": reason,
                    "scorer_id": "deterministic_netops_scorer_v1",
                    "scorer_version": "1.0",
                    "scoring_input_sha256": scoring_input_hash,
                    "scoring_artifact_path": scoring_relative,
                    "scoring_artifact_sha256": scoring_artifact_hash,
                    "contamination_flags": [],
                    "validity_flags": [],
                    "started_at_utc": started,
                    "completed_at_utc": started,
                    "latency_ms": 0,
                }
                row_issues = validate_paired_binary_result_row(row)
                if row_issues:
                    raise RuntimeError(
                        "Generated invalid paired result row: "
                        + "; ".join(row_issues)
                    )
                rows.append(row)
                log_events.append({
                    "event_type": (
                        "cache_reuse" if cached else "model_call_attempt"
                    ),
                    "study_id": study_id,
                    "episode_id": episode_id,
                    "pair_id": task["pair_id"],
                    "task_id": task["task_id"],
                    "condition": condition,
                    "attempt_number": 1,
                    "cache_key_sha256": _sha256_bytes(
                        f"{model_configuration_hash}:{prompt_hash}".encode()
                    ),
                    "provider_request_id": None,
                    "started_at_utc": started,
                    "completed_at_utc": started,
                    "outcome": call_status,
                    "error_type": error_type,
                    "retry_decision": "STOP",
                    "model_calls_used": model_calls_for_episode,
                })

        results_path = output_dir / "raw_results.jsonl"
        results_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        execution_log_path = output_dir / "execution_log.jsonl"
        execution_log_path.write_text(
            "".join(
                json.dumps(event, sort_keys=True) + "\n"
                for event in log_events
            ),
            encoding="utf-8",
        )

        artifact_paths = [
            results_path,
            result_schema_path,
            execution_log_path,
            task_manifest_path,
            transformation_path,
            model_configuration_path,
            *sorted(responses_dir.glob("*.txt")),
            *sorted(scoring_dir.glob("*.json")),
        ]
        artifact_hashes = {
            _relative_to_parent(path, output_dir): _sha256_file(path)
            for path in artifact_paths
        }
        completed_count = sum(row["score"] in (0, 1) for row in rows)
        failed_count = len(rows) - completed_count
        manifest = {
            "status": COMPLETED_STATUS,
            "schema_version": "1.0",
            "adapter_family": self.family,
            "study_id": study_id,
            "started_at_utc": started,
            "completed_at_utc": started,
            "planned_episode_count": len(rows),
            "completed_episode_count": completed_count,
            "failed_episode_count": failed_count,
            "model_calls_used": model_calls_used,
            "results_path": _relative_to_parent(results_path, output_dir),
            "result_schema_path": _relative_to_parent(
                result_schema_path, output_dir
            ),
            "execution_log_path": _relative_to_parent(
                execution_log_path, output_dir
            ),
            "task_manifest_path": _relative_to_parent(
                task_manifest_path, output_dir
            ),
            "transformation_manifest_path": _relative_to_parent(
                transformation_path, output_dir
            ),
            "model_configuration_path": _relative_to_parent(
                model_configuration_path, output_dir
            ),
            "result_schema_id": PAIRED_BINARY_RESULT_SCHEMA_ID,
            "result_schema_version": PAIRED_BINARY_RESULT_SCHEMA_VERSION,
            "execution_mode": "development_rehearsal",
            "artifact_hashes": artifact_hashes,
            "warnings": [
                "DEVELOPMENT_REHEARSAL_ONLY",
                "DETERMINISTIC_LOCAL_STAND_IN_NOT_SCIENTIFIC_EVIDENCE",
            ],
            "preregistration_sha256": _sha256_bytes(
                _canonical_json_bytes(preregistration)
            ),
        }
        manifest_path = output_dir / "execution_manifest.json"
        _write_json(manifest_path, manifest)
        return manifest


def register_builtin_execution_adapters() -> None:
    """Register built-in adapters explicitly and idempotently by family."""
    if SYNTHETIC_PAIRED_ADAPTER_FAMILY not in registered_adapter_families():
        register_adapter(SyntheticPairedLLMBenchmarkAdapter())
