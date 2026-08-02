from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controlled_fault_regime import (
    build_controlled_fault_pair,
    score_controlled_fault_condition,
)
from .model_providers import (
    HostedModelProvider,
    JsonFileCallCache,
    ModelCallRequest,
    OpenAIResponsesProvider,
)
from .netops_generate_validate_repair import (
    TASK_GENERATOR_ID,
    TASK_GENERATOR_VERSION,
    VALIDATOR_ID,
    VALIDATOR_VERSION,
    generate_task,
    validate_configuration,
)


HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY = (
    "hosted_netops_controlled_fault_v1"
)
HOSTED_CONTROLLED_FAULT_ADAPTER_VERSION = "1.0"
SUPPORTED_PROVIDER = "openai_responses"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
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


def _source_prompt(task: dict[str, Any]) -> str:
    state_lines = [
        (
            f"- {name}: admin={state['admin']}, "
            f"mtu={state['mtu']}, vlan={state['vlan']}"
        )
        for name, state in task["initial_state"].items()
    ]
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Current state:\n"
        + "\n".join(state_lines)
        + "\nReturn a valid ordered command sequence. Transient safety "
        "constraints must hold after every command. Generate only required "
        "commands; do not restate preserved settings or add no-op commands. "
        "Use exactly this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>\n"
        "Do not use Markdown or explanatory prose."
    )


def _repair_prompt(
    task: dict[str, Any],
    injected_candidate: str,
    validation: dict[str, Any],
) -> str:
    state_lines = [
        (
            f"- {name}: admin={state['admin']}, "
            f"mtu={state['mtu']}, vlan={state['vlan']}"
        )
        for name, state in task["initial_state"].items()
    ]
    violations = "\n".join(
        f"- {item['code']}: {item['message']}"
        for item in validation["violations"]
    )
    return (
        f"Intent and safety policy: {task['intent']}\n"
        "Initial state:\n"
        + "\n".join(state_lines)
        + "\nThe candidate contains a controlled operational defect and "
        "failed deterministic validation.\n"
        f"Candidate:\n{injected_candidate}\n"
        f"Validation feedback:\n{violations}\n"
        "Return one corrected complete ordered sequence. Preserve all "
        "unrelated settings, satisfy transient constraints after every "
        "command, and do not include no-op or explanatory text. Use exactly "
        "this DSL:\n"
        "interface <name> admin up|down\n"
        "interface <name> mtu <integer>\n"
        "interface <name> vlan <integer>"
    )


def controlled_fault_plan_issues(plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if plan.get("adapter_family") != HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY:
        issues.append("Adapter family is incompatible.")
    if plan.get("execution_mode") != "scientific_pilot":
        issues.append("Only scientific_pilot is supported.")
    if plan.get("conditions") != ["baseline", "guarded"]:
        issues.append("Conditions must be baseline and guarded.")
    if plan.get("model_provider") != SUPPORTED_PROVIDER:
        issues.append("Provider must be openai_responses.")
    if plan.get("maximum_attempts_per_call") != 1:
        issues.append("Exactly one attempt per call is required.")
    if plan.get("reasoning_effort") != "minimal":
        issues.append("reasoning_effort must be minimal.")

    indices = plan.get("task_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in indices
        )
        or len(set(indices)) != len(indices)
    ):
        issues.append("task_indices must be unique positive integers.")
    else:
        expected = len(indices) * 2
        if plan.get("maximum_model_calls") != expected:
            issues.append(
                "maximum_model_calls must equal two calls per task."
            )

    max_output = plan.get("max_output_tokens")
    if (
        not isinstance(max_output, int)
        or isinstance(max_output, bool)
        or not 1 <= max_output <= 2000
    ):
        issues.append("max_output_tokens must be from 1 to 2000.")
    return sorted(set(issues))


class HostedControlledFaultPilot:
    def __init__(
        self,
        provider: HostedModelProvider | None = None,
    ) -> None:
        self.provider = provider

    def _provider_for(
        self,
        execution_dir: Path,
        plan: dict[str, Any],
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
        plan: dict[str, Any],
        execution_dir: Path,
    ) -> dict[str, Any]:
        issues = controlled_fault_plan_issues(plan)
        if issues:
            raise ValueError("; ".join(issues))

        execution_dir.mkdir(parents=True, exist_ok=False)
        prompts_dir = execution_dir / "prompts"
        responses_dir = execution_dir / "responses"
        provider_dir = execution_dir / "provider_calls"
        fault_dir = execution_dir / "faults"
        scoring_dir = execution_dir / "scoring"
        for path in (
            prompts_dir,
            responses_dir,
            provider_dir,
            fault_dir,
            scoring_dir,
        ):
            path.mkdir()

        provider = self._provider_for(execution_dir, plan)
        study_id = str(plan["study_id"])
        rows: list[dict[str, Any]] = []
        model_calls_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        started = datetime.now(timezone.utc)

        for index in plan["task_indices"]:
            task_id = f"task-{index:06d}"
            pair_id = f"pair-{index:06d}"
            task = generate_task(index)

            source_prompt = _source_prompt(task)
            source_prompt_path = (
                prompts_dir / f"{task_id}-source-generation.txt"
            )
            source_prompt_path.write_text(
                source_prompt + "\n",
                encoding="utf-8",
            )

            source_request = ModelCallRequest(
                provider=SUPPORTED_PROVIDER,
                model=str(plan["model_name"]),
                prompt=source_prompt,
                instructions=(
                    "Generate one valid bounded NetOps workflow."
                ),
                temperature=None,
                max_output_tokens=int(plan["max_output_tokens"]),
                reasoning_effort="minimal",
                metadata={
                    "study_id": study_id,
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "stage": "valid_source_generation",
                },
            )

            source_call: dict[str, Any] | None = None
            source_error: str | None = None
            source_candidate: str | None = None
            try:
                result = provider.call(source_request)
                source_call = result.to_dict()
                source_candidate = result.response_text.strip()
                model_calls_used += int(result.attempt_count)
                total_input_tokens += int(result.input_tokens)
                total_output_tokens += int(result.output_tokens)
                total_tokens += int(result.total_tokens)
            except Exception as exc:
                source_error = f"{type(exc).__name__}: {exc}"
                model_calls_used += 1

            _write_json(
                provider_dir / f"{task_id}-source-generation.json",
                {
                    "schema_version": "1.0",
                    "stage": "valid_source_generation",
                    "call": source_call,
                    "terminal_error": source_error,
                },
            )

            if source_candidate is None:
                for condition in ("baseline", "guarded"):
                    rows.append({
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "condition": condition,
                        "call_status": "FAILED",
                        "scoring_status": "NOT_SCORED",
                        "score": None,
                        "score_reason_code": "SOURCE_CALL_FAILED",
                        "fault_class": None,
                        "repair_applied": False,
                        "terminal_error": source_error,
                    })
                continue

            source_response_path = (
                responses_dir / f"{task_id}-valid-source.txt"
            )
            source_response_path.write_text(
                source_candidate + "\n",
                encoding="utf-8",
            )
            source_validation = validate_configuration(
                task,
                source_candidate,
            )
            if not source_validation["valid"]:
                _write_json(
                    scoring_dir / f"{task_id}-source-validation.json",
                    source_validation,
                )
                for condition in ("baseline", "guarded"):
                    rows.append({
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "condition": condition,
                        "call_status": "COMPLETED",
                        "scoring_status": "NOT_SCORED",
                        "score": None,
                        "score_reason_code": "INVALID_SOURCE_CANDIDATE",
                        "fault_class": None,
                        "repair_applied": False,
                        "terminal_error": None,
                    })
                continue

            pair = build_controlled_fault_pair(
                task,
                source_candidate,
            )
            pair["task_id"] = task_id
            pair["pair_id"] = pair_id
            
            injected = pair["shared_injected_candidate"]
            injected_path = (
                responses_dir / f"{task_id}-shared-injected.txt"
            )
            injected_path.write_text(injected + "\n", encoding="utf-8")
            _write_json(
                fault_dir / f"{task_id}-fault.json",
                {
                    "schema_version": "1.0",
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "task_generator_id": TASK_GENERATOR_ID,
                    "task_generator_version": TASK_GENERATOR_VERSION,
                    "validator_id": VALIDATOR_ID,
                    "validator_version": VALIDATOR_VERSION,
                    **pair,
                },
            )

            baseline = score_controlled_fault_condition(
                task,
                pair,
                "baseline",
            )
            _write_json(
                scoring_dir / f"{task_id}-baseline.json",
                baseline,
            )
            rows.append({
                "task_id": task_id,
                "pair_id": pair_id,
                "condition": "baseline",
                "call_status": "COMPLETED",
                "scoring_status": "COMPLETED",
                "score": baseline["score"],
                "score_reason_code": (
                    "VALID_CONFIGURATION"
                    if baseline["score"] == 1
                    else "INVALID_CONFIGURATION"
                ),
                "fault_class": pair["fault_class"],
                "repair_applied": False,
                "terminal_error": None,
            })

            repair_prompt = _repair_prompt(
                task,
                injected,
                pair["injected_validation"],
            )
            repair_prompt_path = (
                prompts_dir / f"{task_id}-guarded-repair.txt"
            )
            repair_prompt_path.write_text(
                repair_prompt + "\n",
                encoding="utf-8",
            )
            repair_request = ModelCallRequest(
                provider=SUPPORTED_PROVIDER,
                model=str(plan["model_name"]),
                prompt=repair_prompt,
                instructions=(
                    "Repair one controlled NetOps workflow defect using "
                    "only the supplied validation feedback."
                ),
                temperature=None,
                max_output_tokens=int(plan["max_output_tokens"]),
                reasoning_effort="minimal",
                metadata={
                    "study_id": study_id,
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "condition": "guarded",
                    "stage": "controlled_fault_repair",
                    "fault_class": pair["fault_class"],
                },
            )

            repair_call: dict[str, Any] | None = None
            repair_error: str | None = None
            repaired_candidate: str | None = None
            try:
                result = provider.call(repair_request)
                repair_call = result.to_dict()
                repaired_candidate = result.response_text.strip()
                model_calls_used += int(result.attempt_count)
                total_input_tokens += int(result.input_tokens)
                total_output_tokens += int(result.output_tokens)
                total_tokens += int(result.total_tokens)
            except Exception as exc:
                repair_error = f"{type(exc).__name__}: {exc}"
                model_calls_used += 1

            _write_json(
                provider_dir / f"{task_id}-guarded-repair.json",
                {
                    "schema_version": "1.0",
                    "stage": "controlled_fault_repair",
                    "fault_class": pair["fault_class"],
                    "call": repair_call,
                    "terminal_error": repair_error,
                },
            )

            if repaired_candidate is None:
                rows.append({
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "condition": "guarded",
                    "call_status": "FAILED",
                    "scoring_status": "NOT_SCORED",
                    "score": None,
                    "score_reason_code": "REPAIR_CALL_FAILED",
                    "fault_class": pair["fault_class"],
                    "repair_applied": True,
                    "terminal_error": repair_error,
                })
                continue

            repaired_path = (
                responses_dir / f"{task_id}-guarded-repaired.txt"
            )
            repaired_path.write_text(
                repaired_candidate + "\n",
                encoding="utf-8",
            )
            guarded = score_controlled_fault_condition(
                task,
                pair,
                "guarded",
                repaired_configuration=repaired_candidate,
            )
            _write_json(
                scoring_dir / f"{task_id}-guarded.json",
                guarded,
            )
            rows.append({
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
                "fault_class": pair["fault_class"],
                "repair_applied": True,
                "terminal_error": None,
            })

            if model_calls_used > int(plan["maximum_model_calls"]):
                raise RuntimeError("Model-call ceiling exceeded.")

        rows_path = execution_dir / "raw_results.jsonl"
        rows_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        pair_ids = sorted({row["pair_id"] for row in rows})
        complete_pairs = 0
        baseline_successes = 0
        guarded_successes = 0
        n_10 = 0
        n_01 = 0
        for pair_id in pair_ids:
            pair_rows = {
                row["condition"]: row
                for row in rows
                if row["pair_id"] == pair_id
            }
            if (
                set(pair_rows) == {"baseline", "guarded"}
                and pair_rows["baseline"]["score"] in (0, 1)
                and pair_rows["guarded"]["score"] in (0, 1)
            ):
                complete_pairs += 1
                baseline = int(pair_rows["baseline"]["score"])
                guarded = int(pair_rows["guarded"]["score"])
                baseline_successes += baseline
                guarded_successes += guarded
                n_10 += int(guarded == 1 and baseline == 0)
                n_01 += int(guarded == 0 and baseline == 1)

        difference = (
            (guarded_successes - baseline_successes) / complete_pairs
            if complete_pairs
            else None
        )
        summary = {
            "schema_version": "1.0",
            "adapter_family": HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY,
            "adapter_version": HOSTED_CONTROLLED_FAULT_ADAPTER_VERSION,
            "study_id": study_id,
            "execution_status": "COMPLETED",
            "task_count": len(plan["task_indices"]),
            "planned_episode_count": len(plan["task_indices"]) * 2,
            "model_calls_used": model_calls_used,
            "maximum_model_calls": plan["maximum_model_calls"],
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "complete_pairs": complete_pairs,
            "baseline_successes": baseline_successes,
            "guarded_successes": guarded_successes,
            "n_10_guarded_only": n_10,
            "n_01_baseline_only": n_01,
            "paired_difference": difference,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(execution_dir / "summary.json", summary)
        return summary
