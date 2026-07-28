from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json


PLACEHOLDERS = (
    "TBD",
    "TODO",
    "PLACEHOLDER",
    "UNKNOWN",
    "<FILL",
    "<REPLACE",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    condition: str
    benchmark: str
    sample_size: int
    task_count: int
    questions_per_task: int
    model: str
    temperature: float
    seed: int
    max_retries: int
    retry_on: list[str]
    output_schema: str
    prompt_template_id: str
    transformation_manifest_id: str | None

    def validate(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.task_count * self.questions_per_task != self.sample_size:
            raise ValueError(
                "task_count * questions_per_task must equal sample_size"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            contains_placeholder(k) or contains_placeholder(v)
            for k, v in value.items()
        )
    if isinstance(value, list):
        return any(contains_placeholder(v) for v in value)
    if isinstance(value, str):
        upper = value.upper()
        return any(token in upper for token in PLACEHOLDERS)
    return False


def validate_preregistration(value: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "programme_id",
        "selected_candidate_id",
        "title",
        "research_question",
        "hypotheses",
        "design",
        "sampling",
        "models",
        "methods",
        "conditions",
        "outcomes",
        "analysis_plan",
        "retry_rules",
        "stopping_rules",
        "deviations_policy",
        "planned_follow_up_candidate_id",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Missing preregistration fields: {sorted(missing)}")
    if contains_placeholder(value):
        raise ValueError("Unresolved placeholder found")

    sampling = value["sampling"]
    if (
        int(sampling["task_count"])
        * int(sampling["questions_per_task"])
        != int(sampling["sample_size"])
    ):
        raise ValueError("Invalid sampling dimensions")

    expected_calls = (
        len(value["methods"])
        * len(value["conditions"])
        * int(sampling["sample_size"])
    )
    if int(value["design"]["planned_evaluations"]) != expected_calls:
        raise ValueError("Invalid planned_evaluations")


def build_preregistration(
    *,
    programme: dict[str, Any],
    selected_study: dict[str, Any],
    finalist_resolution: dict[str, Any],
) -> dict[str, Any]:
    if finalist_resolution["selected_candidate_id"] != "C2":
        raise ValueError("v0.5 default plan supports selected C2")

    value = {
        "schema_version": "0.5.0",
        "programme_id": programme["programme_id"],
        "selected_candidate_id": "C2",
        "title": selected_study["title"],
        "research_question": (
            "Does option-independent structured reasoning reduce "
            "control-adjusted answer-order instability relative to direct "
            "one-letter MCQA answering on NetOps benchmark questions?"
        ),
        "hypotheses": {
            "primary": (
                "Structured reasoning produces lower excess permutation "
                "disagreement than direct MCQA answering."
            ),
            "secondary_non_inferiority": (
                "Structured reasoning does not reduce original-condition "
                "accuracy by more than 0.03."
            ),
        },
        "design": {
            "design_type": "paired controlled confirmatory experiment",
            "planned_evaluations": 3600,
            "pairing_unit": "question-method pair",
            "clustering_unit": "benchmark task",
        },
        "sampling": {
            "benchmark": "6G-Bench",
            "sample_size": 600,
            "task_count": 30,
            "questions_per_task": 20,
            "sample_relation_to_prior_study": "disjoint confirmatory sample",
            "exclusion_rules": [
                "Missing or duplicated option text.",
                "Gold answer absent from option set.",
                "Permutation cannot be mapped semantically.",
                "Question duplicates a frozen prior-study item.",
            ],
        },
        "models": [
            {
                "model_id": "gpt-5-nano",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 600,
            }
        ],
        "methods": [
            {"method_id": "direct", "name": "Direct one-letter MCQA"},
            {
                "method_id": "structured",
                "name": "Structured option-independent reasoning",
            },
        ],
        "conditions": [
            {"condition_id": "original"},
            {"condition_id": "repeat"},
            {"condition_id": "permuted"},
        ],
        "outcomes": {
            "primary": {
                "name": "difference_in_excess_instability",
                "definition": (
                    "Direct excess instability minus structured excess "
                    "instability."
                ),
            },
            "secondary": [
                "accuracy_difference",
                "semantic_permutation_disagreement",
                "unchanged_repeat_disagreement",
            ],
        },
        "analysis_plan": {
            "primary_estimator": (
                "Task-cluster bootstrap difference in excess instability."
            ),
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260730,
            "confidence_level": 0.95,
            "paired_test": "Exact McNemar test",
            "non_inferiority_margin": -0.03,
            "task_sensitivity": "Leave-one-task-out analysis",
        },
        "retry_rules": {
            "max_retries_per_evaluation": 2,
            "retryable_failures": [
                "timeout",
                "rate_limit",
                "transient_server_error",
                "invalid_json",
                "invalid_option_label",
            ],
            "prompt_mutation_on_retry": False,
        },
        "stopping_rules": {
            "early_stopping": False,
            "stop_only_for": [
                "irrecoverable benchmark corruption",
                "provider-wide outage",
                "preregistration integrity failure",
            ],
            "outcome_inspection_before_completion": False,
        },
        "deviations_policy": (
            "Record every deviation before outcome inspection in "
            "deviations.json."
        ),
        "planned_follow_up_candidate_id": finalist_resolution[
            "planned_follow_up_candidate_id"
        ],
    }

    validate_preregistration(value)
    return value


def build_run_specs(value: dict[str, Any]) -> list[RunSpec]:
    sampling = value["sampling"]
    model = value["models"][0]
    seeds = {
        "original": 2026073001,
        "repeat": 2026073002,
        "permuted": 2026073003,
    }
    result: list[RunSpec] = []

    for method in value["methods"]:
        for condition in value["conditions"]:
            method_id = method["method_id"]
            condition_id = condition["condition_id"]

            spec = RunSpec(
                run_id=f"{method_id}-{condition_id}",
                method=method_id,
                condition=condition_id,
                benchmark=sampling["benchmark"],
                sample_size=sampling["sample_size"],
                task_count=sampling["task_count"],
                questions_per_task=sampling["questions_per_task"],
                model=model["model_id"],
                temperature=model["temperature"],
                seed=seeds[condition_id]
                + (100 if method_id == "structured" else 0),
                max_retries=2,
                retry_on=[
                    "timeout",
                    "rate_limit",
                    "transient_server_error",
                    "invalid_json",
                    "invalid_option_label",
                ],
                output_schema="semantic_decision_v1",
                prompt_template_id=(
                    "direct_mcqa_v1"
                    if method_id == "direct"
                    else "structured_semantic_decision_v1"
                ),
                transformation_manifest_id=(
                    "fixed-option-permutation-v1"
                    if condition_id == "permuted"
                    else None
                ),
            )
            spec.validate()
            result.append(spec)

    return result


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
