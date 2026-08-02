from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controlled_fault_experiment_plan import (
    load_experiment_plan,
)
from .controlled_fault_regime import (
    build_controlled_fault_pair,
    score_controlled_fault_condition,
)
from .hosted_controlled_fault_pilot import (
    _repair_prompt,
    _source_prompt,
)
from .model_providers import (
    HostedModelProvider,
    JsonFileCallCache,
    ModelCallRequest,
    OpenAIResponsesProvider,
)
from .netops_generate_validate_repair import (
    generate_task,
    validate_configuration,
)


PLAN_RUNNER_ID = "hosted_controlled_fault_plan_runner_v1"
PLAN_RUNNER_VERSION = "1.0"
PROMPT_PROTOCOL_VERSION = "controlled_fault_prompts_v1"
SUPPORTED_PROVIDER = "openai_responses"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_fingerprint() -> str:
    payload = {
        "runner_id": PLAN_RUNNER_ID,
        "runner_version": PLAN_RUNNER_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "source_prompt": inspect.getsource(_source_prompt),
        "repair_prompt": inspect.getsource(_repair_prompt),
        "build_pair": inspect.getsource(build_controlled_fault_pair),
        "score_condition": inspect.getsource(
            score_controlled_fault_condition
        ),
        "validate_configuration": inspect.getsource(
            validate_configuration
        ),
    }
    return _sha256_text(_canonical_json(payload))


def _new_manifest(
    *,
    plan: dict[str, Any],
    model_name: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "runner_id": PLAN_RUNNER_ID,
        "runner_version": PLAN_RUNNER_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "pair_count": plan["pair_count"],
        "maximum_model_calls": plan["maximum_model_calls"],
        "model_provider": SUPPORTED_PROVIDER,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "maximum_attempts_per_call": 1,
        "reasoning_effort": "minimal",
        "code_fingerprint_sha256": _code_fingerprint(),
        "model_calls_used": 0,
        "created_at_utc": now,
        "updated_at_utc": now,
        "execution_status": "IN_PROGRESS",
    }


def _resume_issues(
    manifest: dict[str, Any],
    *,
    plan: dict[str, Any],
    model_name: str,
    max_output_tokens: int,
) -> list[str]:
    expected = {
        "runner_id": PLAN_RUNNER_ID,
        "runner_version": PLAN_RUNNER_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "pair_count": plan["pair_count"],
        "maximum_model_calls": plan["maximum_model_calls"],
        "model_provider": SUPPORTED_PROVIDER,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "maximum_attempts_per_call": 1,
        "reasoning_effort": "minimal",
        "code_fingerprint_sha256": _code_fingerprint(),
    }
    issues: list[str] = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            issues.append(
                f"Resume mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}."
            )
    return issues


def _checkpoint_path(run_dir: Path, pair_id: str) -> Path:
    return run_dir / "execution" / "checkpoints" / f"{pair_id}.json"


def _pair_artifact_paths(
    execution_dir: Path,
    task_id: str,
) -> dict[str, Path]:
    return {
        "source_prompt": (
            execution_dir / "prompts" / f"{task_id}-source-generation.txt"
        ),
        "repair_prompt": (
            execution_dir / "prompts" / f"{task_id}-guarded-repair.txt"
        ),
        "source_response": (
            execution_dir / "responses" / f"{task_id}-valid-source.txt"
        ),
        "injected_response": (
            execution_dir / "responses" / f"{task_id}-shared-injected.txt"
        ),
        "repair_response": (
            execution_dir / "responses" / f"{task_id}-guarded-repaired.txt"
        ),
        "source_provider": (
            execution_dir
            / "provider_calls"
            / f"{task_id}-source-generation.json"
        ),
        "repair_provider": (
            execution_dir
            / "provider_calls"
            / f"{task_id}-guarded-repair.json"
        ),
        "fault": execution_dir / "faults" / f"{task_id}-fault.json",
        "baseline": (
            execution_dir / "scoring" / f"{task_id}-baseline.json"
        ),
        "guarded": (
            execution_dir / "scoring" / f"{task_id}-guarded.json"
        ),
    }


def _initial_checkpoint(pair_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pair_id": pair_spec["pair_id"],
        "task_id": pair_spec["task_id"],
        "task_index": pair_spec["task_index"],
        "workflow_pattern": pair_spec["workflow_pattern"],
        "fault_class": pair_spec["fault_class"],
        "source_stage": "PENDING",
        "fault_stage": "PENDING",
        "baseline_stage": "PENDING",
        "repair_stage": "PENDING",
        "guarded_stage": "PENDING",
        "rows": [],
        "terminal_error": None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
) -> None:
    checkpoint["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, checkpoint)


def _call_record(
    *,
    stage: str,
    call: dict[str, Any] | None,
    terminal_error: str | None,
    pair_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": stage,
        "pair_id": pair_spec["pair_id"],
        "task_id": pair_spec["task_id"],
        "task_index": pair_spec["task_index"],
        "fault_class": pair_spec["fault_class"],
        "call": call,
        "terminal_error": terminal_error,
    }


class HostedControlledFaultPlanRunner:
    def __init__(
        self,
        provider: HostedModelProvider | None = None,
    ) -> None:
        self.provider = provider

    def _provider_for(
        self,
        execution_dir: Path,
    ) -> HostedModelProvider:
        if self.provider is not None:
            return self.provider
        return OpenAIResponsesProvider(
            cache=JsonFileCallCache(execution_dir / "call_cache"),
            maximum_attempts=1,
            retry_backoff_seconds=0.0,
        )

    def execute(
        self,
        *,
        plan_path: Path,
        run_dir: Path,
        model_name: str,
        max_output_tokens: int,
        resume: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= max_output_tokens <= 2000:
            raise ValueError(
                "max_output_tokens must be between 1 and 2000."
            )

        plan = load_experiment_plan(plan_path.resolve())
        run_dir = run_dir.resolve()
        execution_dir = run_dir / "execution"
        manifest_path = run_dir / "run_manifest.json"

        if resume:
            if not run_dir.exists() or not manifest_path.exists():
                raise ValueError(
                    "Resume requires an existing run and manifest."
                )
            manifest = _read_json(manifest_path)
            issues = _resume_issues(
                manifest,
                plan=plan,
                model_name=model_name,
                max_output_tokens=max_output_tokens,
            )
            if issues:
                raise ValueError(
                    "Resume compatibility check failed:\n- "
                    + "\n- ".join(issues)
                )
        else:
            if run_dir.exists():
                raise FileExistsError(run_dir)
            run_dir.mkdir(parents=True)
            execution_dir.mkdir()
            for directory in (
                "prompts",
                "responses",
                "provider_calls",
                "faults",
                "scoring",
                "checkpoints",
                "call_cache",
            ):
                (execution_dir / directory).mkdir()
            plan_copy = run_dir / "frozen_plan.json"
            plan_copy.write_text(
                plan_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest = _new_manifest(
                plan=plan,
                model_name=model_name,
                max_output_tokens=max_output_tokens,
            )
            _write_json(manifest_path, manifest)

        provider = self._provider_for(execution_dir)

        for pair_spec in plan["pairs"]:
            pair_id = pair_spec["pair_id"]
            task_id = pair_spec["task_id"]
            checkpoint_path = _checkpoint_path(run_dir, pair_id)
            if checkpoint_path.exists():
                checkpoint = _read_json(checkpoint_path)
            else:
                checkpoint = _initial_checkpoint(pair_spec)
                _save_checkpoint(checkpoint_path, checkpoint)

            if (
                checkpoint["baseline_stage"] == "COMPLETED"
                and checkpoint["guarded_stage"] == "COMPLETED"
            ):
                continue
            if checkpoint["source_stage"] in {"FAILED", "INVALID"}:
                continue
            if checkpoint["repair_stage"] == "FAILED":
                continue

            task = generate_task(pair_spec["task_index"])
            paths = _pair_artifact_paths(execution_dir, task_id)

            if checkpoint["source_stage"] != "COMPLETED":
                source_prompt = _source_prompt(task)
                paths["source_prompt"].write_text(
                    source_prompt + "\n",
                    encoding="utf-8",
                )
                request = ModelCallRequest(
                    provider=SUPPORTED_PROVIDER,
                    model=model_name,
                    prompt=source_prompt,
                    instructions=(
                        "Generate one valid bounded NetOps workflow."
                    ),
                    temperature=None,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort="minimal",
                    metadata={
                        "pair_id": pair_id,
                        "task_id": task_id,
                        "task_index": str(pair_spec["task_index"]),
                        "stage": "valid_source_generation",
                        "plan_sha256": plan["plan_sha256"],
                    },
                )
                call_dict: dict[str, Any] | None = None
                error: str | None = None
                try:
                    result = provider.call(request)
                    call_dict = result.to_dict()
                    source_candidate = result.response_text.strip()
                    manifest["model_calls_used"] += int(
                        result.attempt_count
                    )
                except Exception as exc:
                    source_candidate = None
                    error = f"{type(exc).__name__}: {exc}"
                    manifest["model_calls_used"] += 1

                _write_json(
                    paths["source_provider"],
                    _call_record(
                        stage="valid_source_generation",
                        call=call_dict,
                        terminal_error=error,
                        pair_spec=pair_spec,
                    ),
                )
                manifest["updated_at_utc"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                _write_json(manifest_path, manifest)

                if manifest["model_calls_used"] > manifest[
                    "maximum_model_calls"
                ]:
                    raise RuntimeError("Global model-call ceiling exceeded.")

                if source_candidate is None:
                    checkpoint["source_stage"] = "FAILED"
                    checkpoint["terminal_error"] = error
                    checkpoint["rows"] = [
                        {
                            "task_id": task_id,
                            "pair_id": pair_id,
                            "condition": condition,
                            "call_status": "FAILED",
                            "scoring_status": "NOT_SCORED",
                            "score": None,
                            "score_reason_code": "SOURCE_CALL_FAILED",
                            "fault_class": pair_spec["fault_class"],
                            "repair_applied": False,
                            "terminal_error": error,
                        }
                        for condition in ("baseline", "guarded")
                    ]
                    _save_checkpoint(checkpoint_path, checkpoint)
                    continue

                paths["source_response"].write_text(
                    source_candidate + "\n",
                    encoding="utf-8",
                )
                source_validation = validate_configuration(
                    task,
                    source_candidate,
                )
                checkpoint["source_validation"] = source_validation
                checkpoint["source_candidate_sha256"] = _sha256_text(
                    source_candidate
                )
                if not source_validation["valid"]:
                    checkpoint["source_stage"] = "INVALID"
                    checkpoint["rows"] = [
                        {
                            "task_id": task_id,
                            "pair_id": pair_id,
                            "condition": condition,
                            "call_status": "COMPLETED",
                            "scoring_status": "NOT_SCORED",
                            "score": None,
                            "score_reason_code": (
                                "INVALID_SOURCE_CANDIDATE"
                            ),
                            "fault_class": pair_spec["fault_class"],
                            "repair_applied": False,
                            "terminal_error": None,
                        }
                        for condition in ("baseline", "guarded")
                    ]
                    _save_checkpoint(checkpoint_path, checkpoint)
                    continue

                checkpoint["source_stage"] = "COMPLETED"
                _save_checkpoint(checkpoint_path, checkpoint)

            if checkpoint["source_stage"] != "COMPLETED":
                continue

            source_candidate = paths["source_response"].read_text(
                encoding="utf-8"
            ).strip()

            if checkpoint["fault_stage"] != "COMPLETED":
                pair = build_controlled_fault_pair(
                    task,
                    source_candidate,
                    fault_class=pair_spec["fault_class"],
                )
                pair["task_id"] = task_id
                pair["pair_id"] = pair_id
                paths["injected_response"].write_text(
                    pair["shared_injected_candidate"] + "\n",
                    encoding="utf-8",
                )
                _write_json(
                    paths["fault"],
                    {
                        "schema_version": "1.0",
                        "plan_sha256": plan["plan_sha256"],
                        "planned_pair": pair_spec,
                        **pair,
                    },
                )
                checkpoint["fault_stage"] = "COMPLETED"
                checkpoint["shared_injected_candidate_sha256"] = pair[
                    "shared_injected_candidate_sha256"
                ]
                _save_checkpoint(checkpoint_path, checkpoint)
            else:
                pair = _read_json(paths["fault"])

            if checkpoint["baseline_stage"] != "COMPLETED":
                baseline = score_controlled_fault_condition(
                    task,
                    pair,
                    "baseline",
                )
                _write_json(paths["baseline"], baseline)
                checkpoint["baseline_stage"] = "COMPLETED"
                checkpoint["baseline_score"] = baseline["score"]
                _save_checkpoint(checkpoint_path, checkpoint)

            if checkpoint["repair_stage"] != "COMPLETED":
                repair_prompt = _repair_prompt(
                    task,
                    pair["shared_injected_candidate"],
                    pair["injected_validation"],
                )
                paths["repair_prompt"].write_text(
                    repair_prompt + "\n",
                    encoding="utf-8",
                )
                request = ModelCallRequest(
                    provider=SUPPORTED_PROVIDER,
                    model=model_name,
                    prompt=repair_prompt,
                    instructions=(
                        "Repair one controlled NetOps workflow defect "
                        "using only the supplied validation feedback."
                    ),
                    temperature=None,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort="minimal",
                    metadata={
                        "pair_id": pair_id,
                        "task_id": task_id,
                        "task_index": str(pair_spec["task_index"]),
                        "stage": "controlled_fault_repair",
                        "fault_class": pair_spec["fault_class"],
                        "plan_sha256": plan["plan_sha256"],
                    },
                )
                call_dict = None
                error = None
                try:
                    result = provider.call(request)
                    call_dict = result.to_dict()
                    repaired_candidate = result.response_text.strip()
                    manifest["model_calls_used"] += int(
                        result.attempt_count
                    )
                except Exception as exc:
                    repaired_candidate = None
                    error = f"{type(exc).__name__}: {exc}"
                    manifest["model_calls_used"] += 1

                _write_json(
                    paths["repair_provider"],
                    _call_record(
                        stage="controlled_fault_repair",
                        call=call_dict,
                        terminal_error=error,
                        pair_spec=pair_spec,
                    ),
                )
                manifest["updated_at_utc"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                _write_json(manifest_path, manifest)

                if manifest["model_calls_used"] > manifest[
                    "maximum_model_calls"
                ]:
                    raise RuntimeError("Global model-call ceiling exceeded.")

                if repaired_candidate is None:
                    checkpoint["repair_stage"] = "FAILED"
                    checkpoint["terminal_error"] = error
                    checkpoint["guarded_stage"] = "NOT_SCORED"
                    checkpoint["rows"] = [
                        {
                            "task_id": task_id,
                            "pair_id": pair_id,
                            "condition": "baseline",
                            "call_status": "COMPLETED",
                            "scoring_status": "COMPLETED",
                            "score": checkpoint["baseline_score"],
                            "score_reason_code": (
                                "VALID_CONFIGURATION"
                                if checkpoint["baseline_score"] == 1
                                else "INVALID_CONFIGURATION"
                            ),
                            "fault_class": pair_spec["fault_class"],
                            "repair_applied": False,
                            "terminal_error": None,
                        },
                        {
                            "task_id": task_id,
                            "pair_id": pair_id,
                            "condition": "guarded",
                            "call_status": "FAILED",
                            "scoring_status": "NOT_SCORED",
                            "score": None,
                            "score_reason_code": "REPAIR_CALL_FAILED",
                            "fault_class": pair_spec["fault_class"],
                            "repair_applied": True,
                            "terminal_error": error,
                        },
                    ]
                    _save_checkpoint(checkpoint_path, checkpoint)
                    continue

                paths["repair_response"].write_text(
                    repaired_candidate + "\n",
                    encoding="utf-8",
                )
                checkpoint["repair_stage"] = "COMPLETED"
                checkpoint["repaired_candidate_sha256"] = _sha256_text(
                    repaired_candidate
                )
                _save_checkpoint(checkpoint_path, checkpoint)

            if (
                checkpoint["repair_stage"] == "COMPLETED"
                and checkpoint["guarded_stage"] != "COMPLETED"
            ):
                repaired_candidate = paths["repair_response"].read_text(
                    encoding="utf-8"
                ).strip()
                guarded = score_controlled_fault_condition(
                    task,
                    pair,
                    "guarded",
                    repaired_configuration=repaired_candidate,
                )
                _write_json(paths["guarded"], guarded)
                checkpoint["guarded_stage"] = "COMPLETED"
                checkpoint["guarded_score"] = guarded["score"]
                checkpoint["rows"] = [
                    {
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "condition": "baseline",
                        "call_status": "COMPLETED",
                        "scoring_status": "COMPLETED",
                        "score": checkpoint["baseline_score"],
                        "score_reason_code": (
                            "VALID_CONFIGURATION"
                            if checkpoint["baseline_score"] == 1
                            else "INVALID_CONFIGURATION"
                        ),
                        "fault_class": pair_spec["fault_class"],
                        "repair_applied": False,
                        "terminal_error": None,
                    },
                    {
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "condition": "guarded",
                        "call_status": "COMPLETED",
                        "scoring_status": "COMPLETED",
                        "score": guarded["score"],
                        "score_reason_code": (
                            "VALID_CONFIGURATION"
                            if guarded["score"] == 1
                            else "INVALID_CONFIGURATION"
                        ),
                        "fault_class": pair_spec["fault_class"],
                        "repair_applied": True,
                        "terminal_error": None,
                    },
                ]
                _save_checkpoint(checkpoint_path, checkpoint)

        rows: list[dict[str, Any]] = []
        completed_pair_count = 0
        terminal_pair_count = 0
        for pair_spec in plan["pairs"]:
            path = _checkpoint_path(run_dir, pair_spec["pair_id"])
            if not path.exists():
                continue
            checkpoint = _read_json(path)
            rows.extend(checkpoint.get("rows", []))
            if (
                checkpoint.get("baseline_stage") == "COMPLETED"
                and checkpoint.get("guarded_stage") == "COMPLETED"
            ):
                completed_pair_count += 1
            if checkpoint.get("rows"):
                terminal_pair_count += 1

        raw_results_path = execution_dir / "raw_results.jsonl"
        raw_results_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        complete_rows: dict[str, dict[str, int]] = {}
        for row in rows:
            if row.get("score") in (0, 1):
                complete_rows.setdefault(row["pair_id"], {})[
                    row["condition"]
                ] = int(row["score"])

        scientific_pairs = [
            values
            for values in complete_rows.values()
            if set(values) == {"baseline", "guarded"}
        ]
        baseline_successes = sum(
            values["baseline"] for values in scientific_pairs
        )
        guarded_successes = sum(
            values["guarded"] for values in scientific_pairs
        )
        n_10 = sum(
            values["baseline"] == 0 and values["guarded"] == 1
            for values in scientific_pairs
        )
        n_01 = sum(
            values["baseline"] == 1 and values["guarded"] == 0
            for values in scientific_pairs
        )

        manifest["execution_status"] = (
            "COMPLETED"
            if terminal_pair_count == plan["pair_count"]
            else "IN_PROGRESS"
        )
        manifest["updated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_json(manifest_path, manifest)

        summary = {
            "schema_version": "1.0",
            "adapter_family": "hosted_netops_controlled_fault_v1",
            "runner_id": PLAN_RUNNER_ID,
            "runner_version": PLAN_RUNNER_VERSION,
            "study_id": run_dir.name,
            "plan_sha256": plan["plan_sha256"],
            "execution_status": manifest["execution_status"],
            "planned_pair_count": plan["pair_count"],
            "terminal_pair_count": terminal_pair_count,
            "completed_pair_count": completed_pair_count,
            "complete_scientific_pair_count": len(scientific_pairs),
            "model_calls_used": manifest["model_calls_used"],
            "maximum_model_calls": manifest["maximum_model_calls"],
            "baseline_successes": baseline_successes,
            "guarded_successes": guarded_successes,
            "n_10_guarded_only": n_10,
            "n_01_baseline_only": n_01,
            "paired_difference": (
                (guarded_successes - baseline_successes)
                / len(scientific_pairs)
                if scientific_pairs
                else None
            ),
        }
        _write_json(execution_dir / "summary.json", summary)
        return summary
