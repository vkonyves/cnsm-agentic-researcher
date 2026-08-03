from cnsm_agentic.autonomous_research.controlled_fault_final_report import _aggregate_group, _bootstrap_difference_ci, _exact_mcnemar_two_sided

def test_exact_mcnemar_for_final_counts() -> None:
    assert abs(_exact_mcnemar_two_sided(37, 0) - (2 / (2 ** 37))) < 1e-18

def test_bootstrap_is_deterministic() -> None:
    differences = [1] * 37 + [0] * 2
    first = _bootstrap_difference_ci(differences, resamples=10000, seed=7)
    second = _bootstrap_difference_ci(differences, resamples=10000, seed=7)
    assert first == second
    assert first == (34 / 39, 1.0)

def test_group_aggregation() -> None:
    records = [
        {"fault_class": "a", "baseline_score": 0, "guarded_score": 1},
        {"fault_class": "a", "baseline_score": 0, "guarded_score": 0},
        {"fault_class": "b", "baseline_score": 0, "guarded_score": 1},
    ]
    result = _aggregate_group(records, "fault_class")
    assert result[0]["fault_class"] == "a"
    assert result[0]["guarded_successes"] == 1
    assert result[0]["paired_difference"] == 0.5
    assert result[1]["paired_difference"] == 1.0
