import json
from pathlib import Path

import pytest

from cnsm_agentic.autonomous_research.controlled_fault_analysis import analyze_controlled_fault_execution


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_execution(root: Path, pairs: list[tuple[int, int]]) -> Path:
    execution = root / "execution"
    execution.mkdir(parents=True)
    _write_json(execution / "summary.json", {
        "adapter_family": "hosted_netops_controlled_fault_v1",
        "study_id": "analysis-test",
    })
    rows = []
    for index, (baseline_score, guarded_score) in enumerate(pairs, start=1):
        task_id = f"task-{index:06d}"
        pair_id = f"pair-{index:06d}"
        shared = f"shared-{index}"
        rows.extend([
            {"task_id": task_id, "pair_id": pair_id, "condition": "baseline", "scoring_status": "COMPLETED", "score_reason_code": "TEST", "score": baseline_score},
            {"task_id": task_id, "pair_id": pair_id, "condition": "guarded", "scoring_status": "COMPLETED", "score_reason_code": "TEST", "score": guarded_score},
        ])
        _write_json(execution / "faults" / f"{task_id}-fault.json", {
            "fault_class": "test_fault",
            "shared_injected_candidate_sha256": shared,
        })
        _write_json(execution / "scoring" / f"{task_id}-baseline.json", {
            "condition_input_candidate_sha256": shared,
            "final_configuration_sha256": shared,
            "repair_applied": False,
        })
        _write_json(execution / "scoring" / f"{task_id}-guarded.json", {
            "condition_input_candidate_sha256": shared,
            "final_configuration_sha256": f"repaired-{index}",
            "repair_applied": True,
        })
    (execution / "raw_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return execution


def test_analysis_computes_paired_statistics(tmp_path: Path) -> None:
    execution = _make_execution(tmp_path, [(0, 1), (0, 1), (1, 1), (0, 0)])
    result = analyze_controlled_fault_execution(
        execution, bootstrap_resamples=1000, bootstrap_seed=7, persist=True
    )
    assert result["complete_pair_count"] == 4
    assert result["baseline_successes"] == 1
    assert result["guarded_successes"] == 3
    assert result["n_10_guarded_only"] == 2
    assert result["n_01_baseline_only"] == 0
    assert result["paired_difference"] == 0.5
    assert result["exact_mcnemar_p_value"] == 0.5
    assert result["validation_status"] == "PASS"
    assert (tmp_path / "analysis/controlled_fault_analysis.json").exists()


def test_analysis_rejects_mismatched_shared_inputs(tmp_path: Path) -> None:
    execution = _make_execution(tmp_path, [(0, 1)])
    path = execution / "scoring/task-000001-guarded.json"
    guarded = json.loads(path.read_text(encoding="utf-8"))
    guarded["condition_input_candidate_sha256"] = "different"
    _write_json(path, guarded)
    with pytest.raises(ValueError, match="guarded input differs"):
        analyze_controlled_fault_execution(execution)


def test_analysis_keeps_incomplete_pairs_out_of_statistics(tmp_path: Path) -> None:
    execution = _make_execution(tmp_path, [(0, 1)])
    rows_path = execution / "raw_results.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    rows.extend([
        {"task_id": "task-000002", "pair_id": "pair-000002", "condition": "baseline", "scoring_status": "COMPLETED", "score_reason_code": "INVALID_CONFIGURATION", "score": 0},
        {"task_id": "task-000002", "pair_id": "pair-000002", "condition": "guarded", "scoring_status": "NOT_SCORED", "score_reason_code": "REPAIR_CALL_FAILED", "score": None},
    ])
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = analyze_controlled_fault_execution(execution, bootstrap_resamples=100, persist=False)
    assert result["complete_pair_count"] == 1
    assert result["incomplete_pair_count"] == 1
    assert result["paired_difference"] == 1.0


def test_analysis_rejects_wrong_adapter_family(tmp_path: Path) -> None:
    execution = _make_execution(tmp_path, [(0, 1)])
    _write_json(execution / "summary.json", {
        "adapter_family": "hosted_netops_gvr_v1",
        "study_id": "wrong-regime",
    })
    with pytest.raises(ValueError, match="does not belong"):
        analyze_controlled_fault_execution(execution)
