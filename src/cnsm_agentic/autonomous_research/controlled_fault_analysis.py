from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

CONTROLLED_FAULT_ANALYZER_ID = "controlled_fault_paired_analyzer_v1"
CONTROLLED_FAULT_ANALYZER_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exact_mcnemar_p_value(n_10: int, n_01: int) -> float:
    discordant = n_10 + n_01
    if discordant == 0:
        return 1.0
    smaller = min(n_10, n_01)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _paired_bootstrap_interval(
    pairs: list[tuple[int, int]], *, resamples: int, seed: int, alpha: float
) -> tuple[float, float]:
    if not pairs:
        raise ValueError("At least one complete pair is required.")
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")
    rng = random.Random(seed)
    count = len(pairs)
    differences = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(count)] for _ in range(count)]
        differences.append(sum(g - b for b, g in sample) / count)
    differences.sort()
    return _percentile(differences, alpha / 2), _percentile(differences, 1 - alpha / 2)


def _load_rows(execution_dir: Path) -> list[dict[str, Any]]:
    path = execution_dir / "raw_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on raw_results.jsonl line {line_number}.") from exc
    return rows


def analyze_controlled_fault_execution(
    execution_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 7,
    confidence_level: float = 0.95,
    persist: bool = True,
) -> dict[str, Any]:
    execution_dir = execution_dir.resolve()
    summary = _read_json(execution_dir / "summary.json")
    if summary.get("adapter_family") != "hosted_netops_controlled_fault_v1":
        raise ValueError("Execution does not belong to the controlled-fault adapter.")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    issues: list[str] = []
    for row in _load_rows(execution_dir):
        pair_id = row.get("pair_id")
        condition = row.get("condition")
        if not isinstance(pair_id, str) or not pair_id:
            issues.append("A result row has no valid pair_id.")
            continue
        if condition not in {"baseline", "guarded"}:
            issues.append(f"{pair_id} has unsupported condition {condition!r}.")
            continue
        if condition in grouped.setdefault(pair_id, {}):
            issues.append(f"{pair_id} contains duplicate {condition} rows.")
            continue
        grouped[pair_id][condition] = row

    pairs: list[tuple[int, int]] = []
    complete_records = []
    incomplete_records = []
    for pair_id in sorted(grouped):
        conditions = grouped[pair_id]
        if set(conditions) != {"baseline", "guarded"}:
            incomplete_records.append({"pair_id": pair_id, "reason": "MISSING_CONDITION"})
            continue
        baseline_row = conditions["baseline"]
        guarded_row = conditions["guarded"]
        task_id = baseline_row.get("task_id")
        if task_id != guarded_row.get("task_id"):
            issues.append(f"{pair_id} has inconsistent task IDs.")
            continue
        b = baseline_row.get("score")
        g = guarded_row.get("score")
        if b not in (0, 1) or g not in (0, 1):
            incomplete_records.append({
                "pair_id": pair_id,
                "task_id": task_id,
                "reason": "PAIR_NOT_FULLY_SCORED",
                "baseline_reason": baseline_row.get("score_reason_code"),
                "guarded_reason": guarded_row.get("score_reason_code"),
            })
            continue
        if not isinstance(task_id, str) or not task_id:
            issues.append(f"{pair_id} has no valid task_id.")
            continue

        fault_path = execution_dir / "faults" / f"{task_id}-fault.json"
        baseline_path = execution_dir / "scoring" / f"{task_id}-baseline.json"
        guarded_path = execution_dir / "scoring" / f"{task_id}-guarded.json"
        if not all(path.exists() for path in (fault_path, baseline_path, guarded_path)):
            issues.append(f"{pair_id} is missing required fault/scoring artifacts.")
            continue

        fault = _read_json(fault_path)
        baseline = _read_json(baseline_path)
        guarded = _read_json(guarded_path)
        shared = fault.get("shared_injected_candidate_sha256")
        if baseline.get("condition_input_candidate_sha256") != shared:
            issues.append(f"{pair_id} baseline input differs from shared candidate.")
            continue
        if guarded.get("condition_input_candidate_sha256") != shared:
            issues.append(f"{pair_id} guarded input differs from shared candidate.")
            continue
        if baseline.get("repair_applied") is not False:
            issues.append(f"{pair_id} baseline unexpectedly applied repair.")
            continue
        if baseline.get("final_configuration_sha256") != shared:
            issues.append(f"{pair_id} baseline did not preserve injected candidate.")
            continue

        pair = (int(b), int(g))
        pairs.append(pair)
        complete_records.append({
            "pair_id": pair_id,
            "task_id": task_id,
            "fault_class": fault.get("fault_class"),
            "shared_injected_candidate_sha256": shared,
            "baseline_score": pair[0],
            "guarded_score": pair[1],
            "guarded_repair_applied": guarded.get("repair_applied"),
        })

    if issues:
        raise ValueError("Controlled-fault analysis validation failed:\n- " + "\n- ".join(sorted(set(issues))))
    if not pairs:
        raise ValueError("No complete scored pairs are available for scientific analysis.")

    n = len(pairs)
    baseline_successes = sum(b for b, _ in pairs)
    guarded_successes = sum(g for _, g in pairs)
    n_10 = sum(g == 1 and b == 0 for b, g in pairs)
    n_01 = sum(g == 0 and b == 1 for b, g in pairs)
    difference = (guarded_successes - baseline_successes) / n
    alpha = 1.0 - confidence_level
    ci_low, ci_high = _paired_bootstrap_interval(
        pairs, resamples=bootstrap_resamples, seed=bootstrap_seed, alpha=alpha
    )
    result = {
        "schema_version": "1.0",
        "analyzer_id": CONTROLLED_FAULT_ANALYZER_ID,
        "analyzer_version": CONTROLLED_FAULT_ANALYZER_VERSION,
        "adapter_family": summary["adapter_family"],
        "study_id": summary.get("study_id"),
        "complete_pair_count": n,
        "incomplete_pair_count": len(incomplete_records),
        "baseline_successes": baseline_successes,
        "guarded_successes": guarded_successes,
        "baseline_accuracy": baseline_successes / n,
        "guarded_accuracy": guarded_successes / n,
        "paired_difference": difference,
        "n_10_guarded_only": n_10,
        "n_01_baseline_only": n_01,
        "exact_mcnemar_p_value": _exact_mcnemar_p_value(n_10, n_01),
        "bootstrap": {
            "method": "paired_nonparametric_percentile",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "paired_difference_ci_low": ci_low,
            "paired_difference_ci_high": ci_high,
        },
        "complete_pairs": complete_records,
        "incomplete_pairs": incomplete_records,
        "validation_status": "PASS",
    }
    if persist:
        analysis_dir = execution_dir.parent / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        _write_json(analysis_dir / "controlled_fault_analysis.json", result)
    return result
