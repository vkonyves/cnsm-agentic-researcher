from __future__ import annotations

from .models import ExperimentNode, ResearchCandidate, StudyPlan


def build_experiment_dag(candidate: ResearchCandidate) -> list[ExperimentNode]:
    return [
        ExperimentNode("validate-inputs", "validation", (), {"candidate_id": candidate.candidate_id}, ("validation_report.json",), ("sample_hash_matches_preregistration", "zero_duplicate_ids"), "ready"),
        ExperimentNode("baseline-original", "model-evaluation", ("validate-inputs",), {"method": "direct", "condition": "original"}, ("predictions-baseline-original.jsonl",), ("validation_complete", "run_spec_frozen")),
        ExperimentNode("baseline-repeat", "model-evaluation", ("validate-inputs",), {"method": "direct", "condition": "repeat"}, ("predictions-baseline-repeat.jsonl",), ("validation_complete", "run_spec_frozen")),
        ExperimentNode("baseline-permuted", "model-evaluation", ("validate-inputs",), {"method": "direct", "condition": "permuted"}, ("predictions-baseline-permuted.jsonl",), ("validation_complete", "transformation_manifest_frozen")),
        ExperimentNode("candidate-original", "model-evaluation", ("validate-inputs",), {"method": candidate.candidate_id, "condition": "original"}, ("predictions-method-original.jsonl",), ("validation_complete", "run_spec_frozen")),
        ExperimentNode("candidate-repeat", "model-evaluation", ("validate-inputs",), {"method": candidate.candidate_id, "condition": "repeat"}, ("predictions-method-repeat.jsonl",), ("validation_complete", "run_spec_frozen")),
        ExperimentNode("candidate-permuted", "model-evaluation", ("validate-inputs",), {"method": candidate.candidate_id, "condition": "permuted"}, ("predictions-method-permuted.jsonl",), ("validation_complete", "transformation_manifest_frozen")),
        ExperimentNode("primary-analysis", "statistics", ("baseline-original", "baseline-repeat", "baseline-permuted", "candidate-original", "candidate-repeat", "candidate-permuted"), {"cluster_unit": "task", "bootstrap_repetitions": 10000}, ("primary_analysis.json", "task_summary.csv"), ("all_predictions_complete", "analysis_spec_frozen")),
        ExperimentNode("falsification-suite", "robustness", ("primary-analysis",), {"checks": ["independent semantic remapping", "identity transformation", "leave-one-task-out", "duplicate and hash verification"]}, ("falsification_report.json",), ("primary_analysis_complete",)),
        ExperimentNode("freeze-claims", "claim-ledger", ("falsification-suite",), {}, ("claim_ledger.json",), ("falsification_passed",)),
    ]


def build_study_plan(candidate: ResearchCandidate, tournament_result: dict[str, object]) -> StudyPlan:
    plan = StudyPlan(
        study_id=f"study-{candidate.candidate_id.lower()}",
        selected_candidate_id=candidate.candidate_id,
        title=candidate.title,
        research_question=candidate.research_question,
        hypotheses=candidate.hypotheses,
        preregistration_fields={
            "research_question": candidate.research_question,
            "hypotheses": [h.__dict__ for h in candidate.hypotheses],
            "primary_outcome": candidate.primary_outcome,
            "sample_policy": "disjoint sample frozen before model execution",
            "selective_reruns": "prohibited",
            "resume_policy": "process only missing records",
            "cluster_unit": "task",
            "post_hoc_policy": "label all unregistered analyses explicitly",
        },
        experiment_nodes=build_experiment_dag(candidate),
        rejected_candidates=dict(tournament_result["rejected_candidates"]),
        selection_rationale=str(tournament_result["selection_rationale"]),
    )
    plan.validate()
    return plan
