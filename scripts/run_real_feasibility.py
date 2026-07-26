from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    Runner,
    flush_traces,
)
from openai.types.shared import Reasoning

from cnsm_agentic.benchmark_schemas import MCQARecord
from cnsm_agentic.model_io import (
    format_mcqa_prompt,
    parse_mcqa_output,
)
from cnsm_agentic.openai_client import configure_openai_client
from cnsm_agentic.settings import load_settings


DEFAULT_LIMIT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real-model MCQA experiment on a selected "
            "6G-Bench JSONL sample."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Path to the source research-pilot run.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of sample records to process.",
    )

    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(
            "datasets/samples/"
            "6g_bench_feasibility_2_per_task.jsonl"
        ),
        help=(
            "Sample path relative to the source-run directory."
        ),
    )

    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=(
            "Optional label used in the experiment "
            "output-directory name."
        ),
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_records(
    path: Path,
    limit: int,
) -> list[MCQARecord]:
    records: list[MCQARecord] = []
    seen_question_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = MCQARecord.model_validate_json(
                    stripped
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL record at line "
                    f"{line_number}: {exc}"
                ) from exc

            if record.question_id in seen_question_ids:
                raise ValueError(
                    f"Duplicate sample question ID at line "
                    f"{line_number}: {record.question_id}"
                )

            seen_question_ids.add(record.question_id)
            records.append(record)

            if len(records) >= limit:
                break

    return records


def load_prediction_records(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    predictions: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                prediction = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing prediction at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(prediction, dict):
                raise ValueError(
                    f"Prediction at line {line_number} "
                    "is not a JSON object."
                )

            if "question_id" not in prediction:
                raise ValueError(
                    f"Prediction at line {line_number} "
                    "has no question_id."
                )

            question_id = str(
                prediction["question_id"]
            )

            if question_id in seen_question_ids:
                raise ValueError(
                    f"Duplicate existing prediction: "
                    f"{question_id}"
                )

            seen_question_ids.add(question_id)
            predictions.append(prediction)

    return predictions


def append_prediction(
    path: Path,
    prediction: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                prediction,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def extract_usage(
    result: Any,
) -> dict[str, Any]:
    usage = result.context_wrapper.usage

    input_details = getattr(
        usage,
        "input_tokens_details",
        None,
    )

    output_details = getattr(
        usage,
        "output_tokens_details",
        None,
    )

    return {
        "requests": getattr(
            usage,
            "requests",
            None,
        ),
        "input_tokens": getattr(
            usage,
            "input_tokens",
            None,
        ),
        "output_tokens": getattr(
            usage,
            "output_tokens",
            None,
        ),
        "total_tokens": getattr(
            usage,
            "total_tokens",
            None,
        ),
        "cached_input_tokens": getattr(
            input_details,
            "cached_tokens",
            None,
        ),
        "reasoning_tokens": getattr(
            output_details,
            "reasoning_tokens",
            None,
        ),
    }


def build_report(
    *,
    predictions: list[dict[str, Any]],
    requested_record_count: int,
    sample_path: Path,
    source_run: Path,
    model_name: str,
    safe_run_label: str,
    predictions_path: Path,
) -> dict[str, Any]:
    successful = [
        prediction
        for prediction in predictions
        if prediction.get("error") is None
    ]

    parsed_predictions = [
        prediction
        for prediction in successful
        if prediction.get("parsed_option") is not None
    ]

    correct_count = sum(
        prediction.get("is_correct") is True
        for prediction in predictions
    )

    total_input_tokens = sum(
        (
            prediction.get("usage") or {}
        ).get("input_tokens")
        or 0
        for prediction in predictions
    )

    total_output_tokens = sum(
        (
            prediction.get("usage") or {}
        ).get("output_tokens")
        or 0
        for prediction in predictions
    )

    total_tokens = sum(
        (
            prediction.get("usage") or {}
        ).get("total_tokens")
        or 0
        for prediction in predictions
    )

    total_latency = sum(
        float(
            prediction.get(
                "latency_seconds",
                0.0,
            )
            or 0.0
        )
        for prediction in predictions
    )

    return {
        "benchmark": "6G-Bench",
        "model": model_name,
        "run_label": safe_run_label,
        "sample_path": str(
            sample_path.relative_to(
                source_run
            )
        ),
        "created_at_utc": utc_now(),
        "requested_record_count": (
            requested_record_count
        ),
        "record_count": len(predictions),
        "successful_model_call_count": len(
            successful
        ),
        "model_error_count": (
            len(predictions)
            - len(successful)
        ),
        "parsed_count": len(
            parsed_predictions
        ),
        "parse_failure_count": (
            len(successful)
            - len(parsed_predictions)
        ),
        "correct_count": correct_count,
        "accuracy_on_parsed": (
            correct_count
            / len(parsed_predictions)
            if parsed_predictions
            else None
        ),
        "coverage": (
            len(parsed_predictions)
            / len(predictions)
            if predictions
            else 0.0
        ),
        "total_latency_seconds": (
            total_latency
        ),
        "mean_latency_seconds": (
            total_latency
            / len(predictions)
            if predictions
            else None
        ),
        "total_input_tokens": (
            total_input_tokens
        ),
        "total_output_tokens": (
            total_output_tokens
        ),
        "total_tokens": total_tokens,
        "predictions_path": str(
            predictions_path.relative_to(
                source_run
            )
        ),
    }


async def main_async() -> int:
    args = parse_args()

    if args.limit < 1:
        raise ValueError(
            "--limit must be at least 1."
        )

    settings = load_settings()

    if not settings.api_key_present:
        raise RuntimeError(
            "OPENAI_API_KEY is missing."
        )

    configure_openai_client()

    sample_path = (
        args.source_run
        / args.sample
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            f"MCQA sample not found: "
            f"{sample_path}"
        )

    records = load_records(
        path=sample_path,
        limit=args.limit,
    )

    if not records:
        raise RuntimeError(
            "The selected sample contains no records."
        )

    model_name = settings.worker_model

    agent = Agent(
        name="6G-Bench MCQA Feasibility Model",
        model=model_name,
        model_settings=ModelSettings(
            reasoning=Reasoning(
                effort="minimal",
            ),
            verbosity="low",
            max_tokens=256,
        ),
        instructions=(
            "Answer the supplied multiple-choice question. "
            "Return exactly one option label: A, B, C, or D. "
            "Return no explanation, punctuation, or "
            "additional text."
        ),
    )

    run_label = (
        args.run_label
        or f"n{len(records)}"
    )

    safe_run_label = "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "-"
        for character in run_label
    )

    output_directory = (
        args.source_run
        / "experiments"
        / "real_feasibility"
        / (
            f"{model_name.replace('/', '-')}"
            f"-{safe_run_label}"
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_directory
        / "predictions.jsonl"
    )

    report_path = (
        output_directory
        / "report.json"
    )

    existing_predictions = (
        load_prediction_records(
            predictions_path
        )
    )

    sample_question_ids = {
        record.question_id
        for record in records
    }

    existing_question_ids = {
        str(prediction["question_id"])
        for prediction in existing_predictions
    }

    unexpected_existing_ids = (
        existing_question_ids
        - sample_question_ids
    )

    if unexpected_existing_ids:
        preview = sorted(
            unexpected_existing_ids
        )[:10]

        raise ValueError(
            "The existing predictions file contains "
            "question IDs that are not in the selected "
            f"sample: {preview}"
        )

    completed_question_ids = set(
        existing_question_ids
    )

    if completed_question_ids:
        print(
            f"Resuming with "
            f"{len(completed_question_ids)} "
            f"completed questions."
        )

    try:
        for index, record in enumerate(
            records,
            start=1,
        ):
            if (
                record.question_id
                in completed_question_ids
            ):
                print(
                    f"[{index}/{len(records)}] "
                    f"{record.question_id} "
                    "— already completed"
                )
                continue

            print(
                f"[{index}/{len(records)}] "
                f"{record.question_id}"
            )

            prompt: str | None = None
            started_at = utc_now()
            started = time.perf_counter()

            try:
                prompt = format_mcqa_prompt(
                    record
                )

                result = await Runner.run(
                    agent,
                    prompt,
                    max_turns=2,
                    run_config=RunConfig(
                        workflow_name=(
                            "CNSM 6G-Bench "
                            "real feasibility"
                        ),
                        trace_metadata={
                            "stage": (
                                "real_model_feasibility"
                            ),
                            "question_id": (
                                record.question_id
                            ),
                            "task_id": (
                                record.task_id
                            ),
                            "model": model_name,
                            "run_label": (
                                safe_run_label
                            ),
                        },
                    ),
                )

                latency_seconds = (
                    time.perf_counter()
                    - started
                )

                final_output = result.final_output

                raw_output = (
                    str(final_output).strip()
                    if final_output is not None
                    else ""
                )

                parsed = parse_mcqa_output(
                    raw_output=raw_output,
                    valid_options=set(
                        record.options.keys()
                    ),
                )

                is_correct = (
                    parsed.parsed_option
                    == record.correct_option
                    if parsed.parsed_option
                    is not None
                    else None
                )

                prediction = {
                    "question_id": (
                        record.question_id
                    ),
                    "benchmark": (
                        record.benchmark
                    ),
                    "task_id": (
                        record.task_id
                    ),
                    "task_name": (
                        record.task_name
                    ),
                    "gold_option": (
                        record.correct_option
                    ),
                    "raw_output": raw_output,
                    "parsed_option": (
                        parsed.parsed_option
                    ),
                    "parse_status": (
                        parsed.parse_status
                    ),
                    "parser_rule": (
                        parsed.parser_rule
                    ),
                    "is_correct": is_correct,
                    "model": model_name,
                    "run_label": (
                        safe_run_label
                    ),
                    "prompt_sha256": (
                        sha256_text(prompt)
                    ),
                    "started_at_utc": (
                        started_at
                    ),
                    "latency_seconds": (
                        latency_seconds
                    ),
                    "usage": extract_usage(
                        result
                    ),
                    "error_type": None,
                    "error": None,
                }

            except Exception as exc:
                latency_seconds = (
                    time.perf_counter()
                    - started
                )

                prediction = {
                    "question_id": (
                        record.question_id
                    ),
                    "benchmark": (
                        record.benchmark
                    ),
                    "task_id": (
                        record.task_id
                    ),
                    "task_name": (
                        record.task_name
                    ),
                    "gold_option": (
                        record.correct_option
                    ),
                    "raw_output": "",
                    "parsed_option": None,
                    "parse_status": (
                        "model_error"
                    ),
                    "parser_rule": None,
                    "is_correct": None,
                    "model": model_name,
                    "run_label": (
                        safe_run_label
                    ),
                    "prompt_sha256": (
                        sha256_text(prompt)
                        if prompt is not None
                        else None
                    ),
                    "started_at_utc": (
                        started_at
                    ),
                    "latency_seconds": (
                        latency_seconds
                    ),
                    "usage": None,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }

                print(
                    f"  ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            append_prediction(
                predictions_path,
                prediction,
            )

            completed_question_ids.add(
                record.question_id
            )

    finally:
        flush_traces()

    all_predictions = load_prediction_records(
        predictions_path
    )

    final_question_ids = {
        str(prediction["question_id"])
        for prediction in all_predictions
    }

    missing_question_ids = (
        sample_question_ids
        - final_question_ids
    )

    unexpected_final_ids = (
        final_question_ids
        - sample_question_ids
    )

    if unexpected_final_ids:
        raise RuntimeError(
            "Predictions contain IDs outside "
            f"the selected sample: "
            f"{sorted(unexpected_final_ids)[:10]}"
        )

    if missing_question_ids:
        raise RuntimeError(
            "The run ended without predictions for "
            f"{len(missing_question_ids)} questions. "
            "Rerun the same command to resume."
        )

    if len(all_predictions) != len(records):
        raise RuntimeError(
            f"Expected {len(records)} prediction records, "
            f"found {len(all_predictions)}."
        )

    report = build_report(
        predictions=all_predictions,
        requested_record_count=args.limit,
        sample_path=sample_path,
        source_run=args.source_run,
        model_name=model_name,
        safe_run_label=safe_run_label,
        predictions_path=predictions_path,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved predictions: "
        f"{predictions_path}"
    )

    print(
        f"Saved report: "
        f"{report_path}"
    )

    return (
        0
        if (
            report["model_error_count"] == 0
            and report["record_count"]
            == len(records)
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main_async()
        )
    )