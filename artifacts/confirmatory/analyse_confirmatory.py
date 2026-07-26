#!/usr/bin/env python3
"""
Preregistered confirmatory analysis for the CNSM Agentic AI Researcher project.

Primary analysis:
- original vs unchanged-repeat semantic disagreement
- original vs permuted semantic disagreement
- paired instability-control counts
- exact two-sided McNemar test
- absolute excess disagreement
- relative disagreement ratio
- 10,000-repetition task-cluster bootstrap

Secondary analysis:
- condition accuracies
- correctness transitions
- latency and token summaries
- task-level metrics

The permuted answer labels are mapped back to the original semantic option
space by matching normalized option text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


DEFAULT_SOURCE_RUN = Path(
    "runs/20260726T092008Z-research-pilot-llm"
)

DEFAULT_ORIGINAL_SAMPLE = Path(
    "datasets/samples/6g_bench_confirmatory_10_per_task.jsonl"
)

DEFAULT_PERMUTED_SAMPLE = Path(
    "datasets/samples/"
    "6g_bench_confirmatory_10_per_task_permuted.jsonl"
)

DEFAULT_ORIGINAL_RUN = Path(
    "experiments/real_feasibility/"
    "gpt-5-nano-confirmatory-original-n300"
)

DEFAULT_REPEAT_RUN = Path(
    "experiments/real_feasibility/"
    "gpt-5-nano-confirmatory-repeat-n300"
)

DEFAULT_PERMUTED_RUN = Path(
    "experiments/real_feasibility/"
    "gpt-5-nano-confirmatory-permuted-n300"
)

DEFAULT_OUTPUT_DIR = Path(
    "experiments/confirmatory/results"
)

EXPECTED_RECORD_COUNT = 300
EXPECTED_TASK_COUNT = 30
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260730

OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse the three confirmatory MCQA conditions."
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=DEFAULT_SOURCE_RUN,
        help="Root directory of the research-pilot run.",
    )
    parser.add_argument(
        "--original-sample",
        type=Path,
        default=DEFAULT_ORIGINAL_SAMPLE,
        help="Original confirmatory sample relative to source-run.",
    )
    parser.add_argument(
        "--permuted-sample",
        type=Path,
        default=DEFAULT_PERMUTED_SAMPLE,
        help="Permuted confirmatory sample relative to source-run.",
    )
    parser.add_argument(
        "--original-run",
        type=Path,
        default=DEFAULT_ORIGINAL_RUN,
        help="Original prediction directory relative to source-run.",
    )
    parser.add_argument(
        "--repeat-run",
        type=Path,
        default=DEFAULT_REPEAT_RUN,
        help="Repeat prediction directory relative to source-run.",
    )
    parser.add_argument(
        "--permuted-run",
        type=Path,
        default=DEFAULT_PERMUTED_RUN,
        help="Permuted prediction directory relative to source-run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory relative to source-run.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected a JSON object in {path}, line "
                    f"{line_number}; got {type(value).__name__}."
                )

            records.append(value)

    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=False)
            )
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def base_question_id(question_id: str) -> str:
    """Remove the deterministic permutation suffix, when present."""
    return question_id.split("-perm-", 1)[0]


def normalized_option_label(value: Any) -> str:
    label = str(value).strip().upper()

    if len(label) == 1 and label in OPTION_LABELS:
        return label

    match = re.fullmatch(r"(?:OPTION\s*)?([A-Z])[\.\):]?", label)
    if match:
        return match.group(1)

    raise ValueError(f"Invalid MCQA option label: {value!r}")


def normalize_option_text(value: Any) -> str:
    """
    Normalize option text for semantic matching.

    This removes superficial whitespace and leading labels while preserving
    the substantive text.
    """
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"^\s*(?:option\s*)?[A-Z]\s*[\.\):\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.casefold()


def extract_text_from_choice(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "value",
            "answer",
            "option",
            "choice",
            "label_text",
        ):
            if key in value and value[key] is not None:
                return extract_text_from_choice(value[key])

    raise ValueError(f"Cannot extract option text from: {value!r}")


def extract_label_from_choice(
    value: Any,
    fallback_index: int,
) -> str:
    if isinstance(value, dict):
        for key in (
            "label",
            "option_label",
            "key",
            "id",
            "letter",
        ):
            if key in value and value[key] is not None:
                try:
                    return normalized_option_label(value[key])
                except ValueError:
                    pass

    if fallback_index >= len(OPTION_LABELS):
        raise ValueError("Too many answer options.")

    return OPTION_LABELS[fallback_index]


def parse_options_container(container: Any) -> dict[str, str]:
    options: dict[str, str] = {}

    if isinstance(container, dict):
        for raw_label, raw_value in container.items():
            try:
                label = normalized_option_label(raw_label)
            except ValueError:
                if isinstance(raw_value, dict):
                    label = extract_label_from_choice(
                        raw_value,
                        fallback_index=len(options),
                    )
                else:
                    raise

            options[label] = extract_text_from_choice(raw_value)

        return options

    if isinstance(container, list):
        for index, raw_value in enumerate(container):
            label = extract_label_from_choice(raw_value, index)
            options[label] = extract_text_from_choice(raw_value)

        return options

    raise ValueError(
        f"Unsupported options container: {type(container).__name__}"
    )


def extract_options(record: dict[str, Any]) -> dict[str, str]:
    """
    Extract an option-label -> option-text mapping from common MCQA schemas.
    """
    direct_keys = (
        "options",
        "choices",
        "answer_choices",
        "answers",
        "candidates",
        "alternatives",
    )

    for key in direct_keys:
        if key in record and record[key] is not None:
            try:
                options = parse_options_container(record[key])
            except (TypeError, ValueError):
                continue

            if len(options) >= 2:
                return options

    nested_keys = (
        "question",
        "item",
        "example",
        "data",
        "payload",
        "prompt_data",
    )

    for nested_key in nested_keys:
        nested = record.get(nested_key)
        if not isinstance(nested, dict):
            continue

        for key in direct_keys:
            if key in nested and nested[key] is not None:
                try:
                    options = parse_options_container(nested[key])
                except (TypeError, ValueError):
                    continue

                if len(options) >= 2:
                    return options

    # Support schemas with separate option_A, option_B, ... fields.
    field_options: dict[str, str] = {}

    for key, value in record.items():
        match = re.fullmatch(
            r"(?:option|choice|answer)[_\-\s]?([A-Z])",
            key,
            flags=re.IGNORECASE,
        )
        if match and value is not None:
            field_options[match.group(1).upper()] = (
                extract_text_from_choice(value)
            )

    if len(field_options) >= 2:
        return dict(sorted(field_options.items()))

    raise KeyError(
        "Could not locate answer options. "
        f"Available sample keys: {sorted(record.keys())}"
    )


def get_question_id(record: dict[str, Any]) -> str:
    for key in (
        "question_id",
        "sample_id",
        "record_id",
        "item_id",
        "id",
    ):
        value = record.get(key)
        if value is not None:
            return str(value)

    raise KeyError(
        f"No question identifier found. Keys: {sorted(record.keys())}"
    )


def get_task_id(record: dict[str, Any]) -> str:
    for key in (
        "task_id",
        "task",
        "category_id",
        "category",
    ):
        value = record.get(key)
        if value is not None:
            return str(value)

    raise KeyError(
        f"No task identifier found. Keys: {sorted(record.keys())}"
    )


def index_unique(
    records: list[dict[str, Any]],
    *,
    name: str,
    key_function,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for record in records:
        key = key_function(record)

        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")

        result[key] = record

    return result


def validate_prediction_records(
    records: list[dict[str, Any]],
    condition: str,
) -> None:
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"{condition}: expected {EXPECTED_RECORD_COUNT} predictions, "
            f"found {len(records)}."
        )

    for index, record in enumerate(records):
        question_id = get_question_id(record)

        if record.get("parse_status") != "parsed":
            raise ValueError(
                f"{condition}: unparsed prediction at record {index}: "
                f"{question_id}"
            )

        if record.get("error") is not None:
            raise ValueError(
                f"{condition}: model error for {question_id}: "
                f"{record.get('error')}"
            )

        normalized_option_label(record.get("parsed_option"))


def make_permuted_to_original_mapping(
    original_options: dict[str, str],
    permuted_options: dict[str, str],
    *,
    question_id: str,
) -> dict[str, str]:
    """
    Map each permuted option label back to the corresponding original label.

    Mapping is based on normalized option text rather than answer letters.
    """
    original_by_text: dict[str, str] = {}

    for label, text in original_options.items():
        normalized = normalize_option_text(text)

        if normalized in original_by_text:
            raise ValueError(
                f"{question_id}: duplicate normalized option text in "
                f"original sample."
            )

        original_by_text[normalized] = label

    mapping: dict[str, str] = {}

    for permuted_label, text in permuted_options.items():
        normalized = normalize_option_text(text)

        if normalized not in original_by_text:
            raise ValueError(
                f"{question_id}: could not match permuted option "
                f"{permuted_label!r} to an original option.\n"
                f"Permuted text: {text!r}\n"
                f"Original options: {original_options!r}\n"
                f"Permuted options: {permuted_options!r}"
            )

        mapping[permuted_label] = original_by_text[normalized]

    if set(mapping.values()) != set(original_options):
        raise ValueError(
            f"{question_id}: permutation mapping is not bijective: "
            f"{mapping}"
        )

    return mapping


def exact_two_sided_mcnemar(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value.

    Under the null, conditional on b+c discordant pairs, either direction has
    probability 0.5.
    """
    n = b + c

    if n == 0:
        return 1.0

    tail_limit = min(b, c)

    lower_tail = sum(
        math.comb(n, k)
        for k in range(tail_limit + 1)
    ) / (2**n)

    return min(1.0, 2.0 * lower_tail)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def percentile_interval(
    values: list[float],
) -> dict[str, float] | None:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )

    if finite.size == 0:
        return None

    lower, median, upper = np.quantile(
        finite,
        [0.025, 0.5, 0.975],
    )

    return {
        "lower_2_5_percent": float(lower),
        "median": float(median),
        "upper_97_5_percent": float(upper),
        "finite_repetition_count": int(finite.size),
    }


def correctness_transition(
    original_correct: bool,
    comparison_correct: bool,
) -> str:
    if original_correct and comparison_correct:
        return "correct_to_correct"
    if original_correct and not comparison_correct:
        return "correct_to_incorrect"
    if not original_correct and comparison_correct:
        return "incorrect_to_correct"
    return "incorrect_to_incorrect"


def usage_value(record: dict[str, Any], key: str) -> int:
    usage = record.get("usage")

    if not isinstance(usage, dict):
        return 0

    value = usage.get(key, 0)
    return int(value or 0)


def condition_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    correct = sum(bool(record.get("is_correct")) for record in records)
    latencies = [
        float(record["latency_seconds"])
        for record in records
        if record.get("latency_seconds") is not None
    ]

    return {
        "record_count": len(records),
        "correct_count": correct,
        "accuracy": correct / len(records),
        "total_latency_seconds": sum(latencies),
        "mean_latency_seconds": mean(latencies),
        "total_input_tokens": sum(
            usage_value(record, "input_tokens")
            for record in records
        ),
        "total_output_tokens": sum(
            usage_value(record, "output_tokens")
            for record in records
        ),
        "total_tokens": sum(
            usage_value(record, "total_tokens")
            for record in records
        ),
    }


def bootstrap_task_clusters(
    item_rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in item_rows:
        rows_by_task[row["task_id"]].append(row)

    task_ids = sorted(rows_by_task)

    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} tasks, found {len(task_ids)}."
        )

    rng = np.random.default_rng(seed)
    distribution: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        sampled_tasks = rng.choice(
            task_ids,
            size=len(task_ids),
            replace=True,
        )

        sampled_rows: list[dict[str, Any]] = []

        for task_id in sampled_tasks:
            sampled_rows.extend(rows_by_task[str(task_id)])

        n = len(sampled_rows)

        repeat_rate = sum(
            row["repeat_disagree"]
            for row in sampled_rows
        ) / n

        permutation_rate = sum(
            row["permutation_disagree"]
            for row in sampled_rows
        ) / n

        excess = permutation_rate - repeat_rate
        ratio = safe_ratio(permutation_rate, repeat_rate)

        distribution.append(
            {
                "repetition": repetition,
                "repeat_disagreement_rate": repeat_rate,
                "permutation_disagreement_rate": permutation_rate,
                "absolute_excess_disagreement": excess,
                "relative_disagreement_ratio": ratio,
            }
        )

    intervals = {
        "repetitions": repetitions,
        "seed": seed,
        "repeat_disagreement_rate": percentile_interval(
            [
                row["repeat_disagreement_rate"]
                for row in distribution
            ]
        ),
        "permutation_disagreement_rate": percentile_interval(
            [
                row["permutation_disagreement_rate"]
                for row in distribution
            ]
        ),
        "absolute_excess_disagreement": percentile_interval(
            [
                row["absolute_excess_disagreement"]
                for row in distribution
            ]
        ),
        "relative_disagreement_ratio": percentile_interval(
            [
                float(row["relative_disagreement_ratio"])
                for row in distribution
                if row["relative_disagreement_ratio"] is not None
            ]
        ),
    }

    return distribution, intervals


def build_task_summaries(
    item_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in item_rows:
        rows_by_task[row["task_id"]].append(row)

    summaries: list[dict[str, Any]] = []

    for task_id in sorted(rows_by_task):
        rows = rows_by_task[task_id]
        count = len(rows)

        repeat_count = sum(row["repeat_disagree"] for row in rows)
        permutation_count = sum(
            row["permutation_disagree"] for row in rows
        )

        summaries.append(
            {
                "task_id": task_id,
                "task_name": rows[0]["task_name"],
                "record_count": count,
                "original_correct_count": sum(
                    row["original_is_correct"] for row in rows
                ),
                "repeat_correct_count": sum(
                    row["repeat_is_correct"] for row in rows
                ),
                "permuted_correct_count": sum(
                    row["permuted_is_correct"] for row in rows
                ),
                "original_accuracy": sum(
                    row["original_is_correct"] for row in rows
                ) / count,
                "repeat_accuracy": sum(
                    row["repeat_is_correct"] for row in rows
                ) / count,
                "permuted_accuracy": sum(
                    row["permuted_is_correct"] for row in rows
                ) / count,
                "repeat_disagreement_count": repeat_count,
                "permutation_disagreement_count": permutation_count,
                "repeat_disagreement_rate": repeat_count / count,
                "permutation_disagreement_rate": (
                    permutation_count / count
                ),
                "absolute_excess_disagreement": (
                    permutation_count - repeat_count
                ) / count,
                "permutation_only_count": sum(
                    row["permutation_disagree"]
                    and not row["repeat_disagree"]
                    for row in rows
                ),
                "repeat_only_count": sum(
                    row["repeat_disagree"]
                    and not row["permutation_disagree"]
                    for row in rows
                ),
            }
        )

    return summaries


def build_leave_one_task_out(
    item_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_ids = sorted({row["task_id"] for row in item_rows})
    results: list[dict[str, Any]] = []

    for omitted_task in task_ids:
        subset = [
            row
            for row in item_rows
            if row["task_id"] != omitted_task
        ]

        record_count = len(subset)

        repeat_rate = sum(
            row["repeat_disagree"]
            for row in subset
        ) / record_count

        permutation_rate = sum(
            row["permutation_disagree"]
            for row in subset
        ) / record_count

        absolute_excess = permutation_rate - repeat_rate

        permutation_only = sum(
            row["permutation_disagree"]
            and not row["repeat_disagree"]
            for row in subset
        )

        repeat_only = sum(
            row["repeat_disagree"]
            and not row["permutation_disagree"]
            for row in subset
        )

        results.append(
            {
                "omitted_task": omitted_task,
                "record_count": record_count,
                "repeat_disagreement_rate": repeat_rate,
                "permutation_disagreement_rate": permutation_rate,
                "absolute_excess_disagreement": absolute_excess,
                "permutation_only_count": permutation_only,
                "repeat_only_count": repeat_only,
            }
        )

    minimum_row = min(
        results,
        key=lambda row: row["absolute_excess_disagreement"],
    )
    maximum_row = max(
        results,
        key=lambda row: row["absolute_excess_disagreement"],
    )

    summary = {
        "analysis_status": "post_hoc_robustness",
        "task_count": len(task_ids),
        "minimum_absolute_excess_disagreement": (
            minimum_row["absolute_excess_disagreement"]
        ),
        "minimum_when_omitting_task": (
            minimum_row["omitted_task"]
        ),
        "maximum_absolute_excess_disagreement": (
            maximum_row["absolute_excess_disagreement"]
        ),
        "maximum_when_omitting_task": (
            maximum_row["omitted_task"]
        ),
        "all_excess_disagreement_values_positive": all(
            row["absolute_excess_disagreement"] > 0
            for row in results
        ),
    }

    return results, summary


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_ci(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "undefined"

    return (
        f"[{interval['lower_2_5_percent']:.4f}, "
        f"{interval['upper_97_5_percent']:.4f}]"
    )


def build_markdown_report(summary: dict[str, Any]) -> str:
    primary = summary["primary_analysis"]
    conditions = summary["condition_summaries"]
    bootstrap = summary["task_cluster_bootstrap"]
    leave_one_out = summary["leave_one_task_out"]
    paired = primary["paired_instability_control_counts"]

    ratio = primary["relative_disagreement_ratio"]
    ratio_text = "undefined" if ratio is None else f"{ratio:.4f}"

    lines = [
        "# Confirmatory analysis report",
        "",
        "## Confirmatory dataset",
        "",
        f"- Questions: {summary['record_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- Questions per task: {summary['questions_per_task']}",
        "",
        "## Mapping validation",
        "",
        (
            "- Identity permutations found: "
            f"{summary['validation']['identity_permutation_count']}"
        ),
        "- Gold-answer mappings: verified for all records",
        "- Repeat gold answers: verified against original records",
        (
            "- Semantic mapping method: "
            f"{summary['validation']['semantic_mapping_method']}"
        ),
        "",
        "## Condition accuracy",
        "",
        "| Condition | Correct | Accuracy |",
        "|---|---:|---:|",
        (
            f"| Original | {conditions['original']['correct_count']} | "
            f"{format_percent(conditions['original']['accuracy'])} |"
        ),
        (
            f"| Unchanged repeat | "
            f"{conditions['repeat']['correct_count']} | "
            f"{format_percent(conditions['repeat']['accuracy'])} |"
        ),
        (
            f"| Permuted | {conditions['permuted']['correct_count']} | "
            f"{format_percent(conditions['permuted']['accuracy'])} |"
        ),
        "",
        "## Primary semantic-disagreement analysis",
        "",
        (
            f"- Original vs repeat disagreement: "
            f"{primary['repeat_disagreement_count']}/"
            f"{summary['record_count']} = "
            f"{format_percent(primary['repeat_disagreement_rate'])}"
        ),
        (
            f"- Original vs permutation disagreement: "
            f"{primary['permutation_disagreement_count']}/"
            f"{summary['record_count']} = "
            f"{format_percent(primary['permutation_disagreement_rate'])}"
        ),
        (
            f"- Absolute excess disagreement: "
            f"{primary['absolute_excess_disagreement']:.4f} "
            f"({100 * primary['absolute_excess_disagreement']:.2f} "
            "percentage points)"
        ),
        f"- Relative disagreement ratio: {ratio_text}",
        "",
        "## Paired instability-control counts",
        "",
        f"- Both agree: {paired['both_agree']}",
        f"- Both disagree: {paired['both_disagree']}",
        (
            f"- Permutation only disagrees: "
            f"{paired['permutation_only_disagrees']}"
        ),
        (
            f"- Repeat only disagrees: "
            f"{paired['repeat_only_disagrees']}"
        ),
        "",
        "## Exact McNemar test",
        "",
        (
            f"- Discordant pair counts: "
            f"b={primary['mcnemar_b_permutation_only']}, "
            f"c={primary['mcnemar_c_repeat_only']}"
        ),
        (
            f"- Exact two-sided p-value: "
            f"{primary['exact_two_sided_mcnemar_p_value']:.12g}"
        ),
        "",
        "## Task-cluster bootstrap",
        "",
        f"- Repetitions: {bootstrap['repetitions']}",
        f"- Seed: {bootstrap['seed']}",
        (
            "- 95% percentile interval, repeat disagreement rate: "
            f"{format_ci(bootstrap['repeat_disagreement_rate'])}"
        ),
        (
            "- 95% percentile interval, permutation disagreement rate: "
            f"{format_ci(bootstrap['permutation_disagreement_rate'])}"
        ),
        (
            "- 95% percentile interval, absolute excess disagreement: "
            f"{format_ci(bootstrap['absolute_excess_disagreement'])}"
        ),
        (
            "- 95% percentile interval, relative disagreement ratio: "
            f"{format_ci(bootstrap['relative_disagreement_ratio'])}"
        ),
        "",
        "## Leave-one-task-out robustness analysis",
        "",
        (
            "This analysis was conducted post hoc as a robustness "
            "check."
        ),
        "",
        (
            f"- Tasks omitted in turn: "
            f"{leave_one_out['task_count']}"
        ),
        (
            f"- Minimum excess disagreement: "
            f"{100 * leave_one_out['minimum_absolute_excess_disagreement']:.2f} "
            "percentage points"
        ),
        (
            f"- Minimum obtained when omitting: "
            f"{leave_one_out['minimum_when_omitting_task']}"
        ),
        (
            f"- Maximum excess disagreement: "
            f"{100 * leave_one_out['maximum_absolute_excess_disagreement']:.2f} "
            "percentage points"
        ),
        (
            f"- Maximum obtained when omitting: "
            f"{leave_one_out['maximum_when_omitting_task']}"
        ),
        (
            "- Positive excess after every task omission: "
            f"{'yes' if leave_one_out['all_excess_disagreement_values_positive'] else 'no'}"
        ),
        "",
        (
            "The confirmatory effect was therefore not attributable "
            "to any single 6G-Bench task."
        ),
        "",
        "## Correctness transitions",
        "",
        "### Original to unchanged repeat",
        "",
    ]

    for key, value in summary[
        "correctness_transitions"
    ]["original_to_repeat"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")

    lines.extend(
        [
            "",
            "### Original to permutation",
            "",
        ]
    )

    for key, value in summary[
        "correctness_transitions"
    ]["original_to_permutation"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The preregistered primary comparison concerns semantic "
                "answer disagreement relative to the unchanged-repeat "
                "instability control. Aggregate accuracy differences are "
                "secondary and do not by themselves measure permutation "
                "sensitivity."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()

    original_sample_path = source_run / args.original_sample
    permuted_sample_path = source_run / args.permuted_sample

    original_predictions_path = (
        source_run / args.original_run / "predictions.jsonl"
    )
    repeat_predictions_path = (
        source_run / args.repeat_run / "predictions.jsonl"
    )
    permuted_predictions_path = (
        source_run / args.permuted_run / "predictions.jsonl"
    )

    output_dir = source_run / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_samples = read_jsonl(original_sample_path)
    permuted_samples = read_jsonl(permuted_sample_path)

    original_predictions = read_jsonl(original_predictions_path)
    repeat_predictions = read_jsonl(repeat_predictions_path)
    permuted_predictions = read_jsonl(permuted_predictions_path)

    validate_prediction_records(
        original_predictions,
        "original",
    )
    validate_prediction_records(
        repeat_predictions,
        "repeat",
    )
    validate_prediction_records(
        permuted_predictions,
        "permuted",
    )

    if len(original_samples) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RECORD_COUNT} original sample records, "
            f"found {len(original_samples)}."
        )

    if len(permuted_samples) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RECORD_COUNT} permuted sample records, "
            f"found {len(permuted_samples)}."
        )

    original_sample_by_id = index_unique(
        original_samples,
        name="original sample",
        key_function=lambda record: base_question_id(
            get_question_id(record)
        ),
    )

    permuted_sample_by_id = index_unique(
        permuted_samples,
        name="permuted sample",
        key_function=lambda record: base_question_id(
            get_question_id(record)
        ),
    )

    original_prediction_by_id = index_unique(
        original_predictions,
        name="original prediction",
        key_function=lambda record: base_question_id(
            get_question_id(record)
        ),
    )

    repeat_prediction_by_id = index_unique(
        repeat_predictions,
        name="repeat prediction",
        key_function=lambda record: base_question_id(
            get_question_id(record)
        ),
    )

    permuted_prediction_by_id = index_unique(
        permuted_predictions,
        name="permuted prediction",
        key_function=lambda record: base_question_id(
            get_question_id(record)
        ),
    )

    expected_ids = set(original_sample_by_id)

    collections = {
        "permuted samples": set(permuted_sample_by_id),
        "original predictions": set(original_prediction_by_id),
        "repeat predictions": set(repeat_prediction_by_id),
        "permuted predictions": set(permuted_prediction_by_id),
    }

    for name, observed_ids in collections.items():
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)

        if missing or unexpected:
            raise ValueError(
                f"ID mismatch for {name}. "
                f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    item_rows: list[dict[str, Any]] = []
    identity_mapping_count = 0

    for question_id in sorted(expected_ids):
        original_sample = original_sample_by_id[question_id]
        permuted_sample = permuted_sample_by_id[question_id]

        original_prediction = original_prediction_by_id[question_id]
        repeat_prediction = repeat_prediction_by_id[question_id]
        permuted_prediction = permuted_prediction_by_id[question_id]

        original_options = extract_options(original_sample)
        permuted_options = extract_options(permuted_sample)

        permuted_to_original = make_permuted_to_original_mapping(
            original_options,
            permuted_options,
            question_id=question_id,
        )

        is_identity_mapping = all(
            permuted_label == original_label
            for permuted_label, original_label
            in permuted_to_original.items()
        )
        if is_identity_mapping:
            identity_mapping_count += 1

        original_gold = normalized_option_label(
            original_prediction["gold_option"]
        )
        repeat_gold = normalized_option_label(
            repeat_prediction["gold_option"]
        )
        permuted_gold = normalized_option_label(
            permuted_prediction["gold_option"]
        )

        if repeat_gold != original_gold:
            raise ValueError(
                f"{question_id}: inconsistent repeat gold answer: "
                f"original={original_gold}, repeat={repeat_gold}"
            )

        if permuted_gold not in permuted_to_original:
            raise ValueError(
                f"{question_id}: permuted gold answer "
                f"{permuted_gold!r} is not in the option mapping."
            )

        mapped_permuted_gold = permuted_to_original[permuted_gold]

        if mapped_permuted_gold != original_gold:
            raise ValueError(
                f"{question_id}: inconsistent gold-answer mapping: "
                f"original={original_gold}, "
                f"permuted={permuted_gold}, "
                f"mapped={mapped_permuted_gold}"
            )

        original_answer = normalized_option_label(
            original_prediction["parsed_option"]
        )
        repeat_answer = normalized_option_label(
            repeat_prediction["parsed_option"]
        )
        permuted_raw_answer = normalized_option_label(
            permuted_prediction["parsed_option"]
        )

        if permuted_raw_answer not in permuted_to_original:
            raise ValueError(
                f"{question_id}: predicted permuted answer "
                f"{permuted_raw_answer!r} is not in the option mapping."
            )

        permuted_semantic_answer = permuted_to_original[
            permuted_raw_answer
        ]

        original_task_id = get_task_id(original_prediction)
        repeat_task_id = get_task_id(repeat_prediction)
        permuted_task_id = get_task_id(permuted_prediction)

        if len(
            {
                original_task_id,
                repeat_task_id,
                permuted_task_id,
            }
        ) != 1:
            raise ValueError(
                f"{question_id}: task ID mismatch across conditions."
            )

        repeat_disagree = original_answer != repeat_answer
        permutation_disagree = (
            original_answer != permuted_semantic_answer
        )

        item_rows.append(
            {
                "question_id": question_id,
                "permuted_question_id": get_question_id(
                    permuted_prediction
                ),
                "task_id": original_task_id,
                "task_name": original_prediction.get(
                    "task_name",
                    "",
                ),
                "original_answer": original_answer,
                "repeat_answer": repeat_answer,
                "permuted_raw_answer": permuted_raw_answer,
                "permuted_semantic_answer": (
                    permuted_semantic_answer
                ),
                "permuted_to_original_mapping": (
                    permuted_to_original
                ),
                "is_identity_mapping": is_identity_mapping,
                "original_gold_answer": original_gold,
                "repeat_gold_answer": repeat_gold,
                "permuted_raw_gold_answer": permuted_gold,
                "permuted_semantic_gold_answer": mapped_permuted_gold,
                "gold_mapping_consistent": True,
                "repeat_disagree": repeat_disagree,
                "permutation_disagree": permutation_disagree,
                "original_is_correct": bool(
                    original_prediction["is_correct"]
                ),
                "repeat_is_correct": bool(
                    repeat_prediction["is_correct"]
                ),
                "permuted_is_correct": bool(
                    permuted_prediction["is_correct"]
                ),
                "original_to_repeat_correctness_transition": (
                    correctness_transition(
                        bool(original_prediction["is_correct"]),
                        bool(repeat_prediction["is_correct"]),
                    )
                ),
                "original_to_permutation_correctness_transition": (
                    correctness_transition(
                        bool(original_prediction["is_correct"]),
                        bool(permuted_prediction["is_correct"]),
                    )
                ),
                "original_latency_seconds": float(
                    original_prediction["latency_seconds"]
                ),
                "repeat_latency_seconds": float(
                    repeat_prediction["latency_seconds"]
                ),
                "permuted_latency_seconds": float(
                    permuted_prediction["latency_seconds"]
                ),
                "original_prompt_sha256": original_prediction.get(
                    "prompt_sha256"
                ),
                "repeat_prompt_sha256": repeat_prediction.get(
                    "prompt_sha256"
                ),
                "permuted_prompt_sha256": permuted_prediction.get(
                    "prompt_sha256"
                ),
            }
        )

    if identity_mapping_count != 0:
        raise ValueError(
            f"Expected zero identity permutations, "
            f"found {identity_mapping_count}."
        )

    task_ids = sorted({row["task_id"] for row in item_rows})

    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} task IDs, "
            f"found {len(task_ids)}."
        )

    task_counts = Counter(row["task_id"] for row in item_rows)

    if set(task_counts.values()) != {10}:
        raise ValueError(
            f"Expected 10 questions per task. Counts: {task_counts}"
        )

    repeat_disagreement_count = sum(
        row["repeat_disagree"] for row in item_rows
    )
    permutation_disagreement_count = sum(
        row["permutation_disagree"] for row in item_rows
    )

    record_count = len(item_rows)

    repeat_disagreement_rate = (
        repeat_disagreement_count / record_count
    )
    permutation_disagreement_rate = (
        permutation_disagreement_count / record_count
    )

    absolute_excess = (
        permutation_disagreement_rate
        - repeat_disagreement_rate
    )
    relative_ratio = safe_ratio(
        permutation_disagreement_rate,
        repeat_disagreement_rate,
    )

    both_agree = sum(
        not row["repeat_disagree"]
        and not row["permutation_disagree"]
        for row in item_rows
    )
    both_disagree = sum(
        row["repeat_disagree"]
        and row["permutation_disagree"]
        for row in item_rows
    )
    permutation_only = sum(
        not row["repeat_disagree"]
        and row["permutation_disagree"]
        for row in item_rows
    )
    repeat_only = sum(
        row["repeat_disagree"]
        and not row["permutation_disagree"]
        for row in item_rows
    )

    mcnemar_p = exact_two_sided_mcnemar(
        permutation_only,
        repeat_only,
    )

    bootstrap_rows, bootstrap_intervals = (
        bootstrap_task_clusters(
            item_rows,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
    )

    task_summaries = build_task_summaries(item_rows)

    leave_one_task_out_rows, leave_one_task_out_summary = (
        build_leave_one_task_out(item_rows)
    )

    original_to_repeat_transitions = Counter(
        row["original_to_repeat_correctness_transition"]
        for row in item_rows
    )
    original_to_permutation_transitions = Counter(
        row["original_to_permutation_correctness_transition"]
        for row in item_rows
    )

    summary = {
        "analysis_name": "CNSM 2026 confirmatory permutation analysis",
        "source_run": str(source_run),
        "record_count": record_count,
        "task_count": len(task_ids),
        "questions_per_task": 10,
        "validation": {
            "identity_permutation_count": identity_mapping_count,
            "expected_identity_permutation_count": 0,
            "gold_mapping_consistency_checked": True,
            "repeat_gold_consistency_checked": True,
            "semantic_mapping_method": (
                "normalized option-text matching"
            ),
        },
        "condition_summaries": {
            "original": condition_summary(original_predictions),
            "repeat": condition_summary(repeat_predictions),
            "permuted": condition_summary(permuted_predictions),
        },
        "primary_analysis": {
            "repeat_disagreement_count": (
                repeat_disagreement_count
            ),
            "repeat_disagreement_rate": (
                repeat_disagreement_rate
            ),
            "permutation_disagreement_count": (
                permutation_disagreement_count
            ),
            "permutation_disagreement_rate": (
                permutation_disagreement_rate
            ),
            "absolute_excess_disagreement": absolute_excess,
            "absolute_excess_percentage_points": (
                100.0 * absolute_excess
            ),
            "relative_disagreement_ratio": relative_ratio,
            "paired_instability_control_counts": {
                "both_agree": both_agree,
                "both_disagree": both_disagree,
                "permutation_only_disagrees": permutation_only,
                "repeat_only_disagrees": repeat_only,
            },
            "mcnemar_b_permutation_only": permutation_only,
            "mcnemar_c_repeat_only": repeat_only,
            "exact_two_sided_mcnemar_p_value": mcnemar_p,
        },
        "task_cluster_bootstrap": bootstrap_intervals,
        "leave_one_task_out": leave_one_task_out_summary,
        "correctness_transitions": {
            "original_to_repeat": dict(
                sorted(original_to_repeat_transitions.items())
            ),
            "original_to_permutation": dict(
                sorted(
                    original_to_permutation_transitions.items()
                )
            ),
        },
        "input_artifacts": {
            "original_sample": {
                "path": str(original_sample_path),
                "sha256": sha256_file(original_sample_path),
            },
            "permuted_sample": {
                "path": str(permuted_sample_path),
                "sha256": sha256_file(permuted_sample_path),
            },
            "original_predictions": {
                "path": str(original_predictions_path),
                "sha256": sha256_file(
                    original_predictions_path
                ),
            },
            "repeat_predictions": {
                "path": str(repeat_predictions_path),
                "sha256": sha256_file(
                    repeat_predictions_path
                ),
            },
            "permuted_predictions": {
                "path": str(permuted_predictions_path),
                "sha256": sha256_file(
                    permuted_predictions_path
                ),
            },
        },
    }

    write_jsonl(
        output_dir / "confirmatory_item_level.jsonl",
        item_rows,
    )
    write_json(
        output_dir / "confirmatory_summary.json",
        summary,
    )
    write_csv(
        output_dir / "task_level_summary.csv",
        task_summaries,
    )
    write_csv(
        output_dir / "leave_one_task_out.csv",
        leave_one_task_out_rows,
    )
    write_csv(
        output_dir / "bootstrap_distribution.csv",
        bootstrap_rows,
    )

    report = build_markdown_report(summary)

    report_path = output_dir / "confirmatory_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"Saved item-level results: {output_dir / 'confirmatory_item_level.jsonl'}")
    print(f"Saved summary: {output_dir / 'confirmatory_summary.json'}")
    print(f"Saved task table: {output_dir / 'task_level_summary.csv'}")
    print(f"Saved bootstrap: {output_dir / 'bootstrap_distribution.csv'}")
    print(f"Saved leave-one-task-out table: "f"{output_dir / 'leave_one_task_out.csv'}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()