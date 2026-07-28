from __future__ import annotations

from .models import Hypothesis, ResearchCandidate, ResearchProgramme


def _h(identifier: str, statement: str, kind: str, outcome: str, null: str, falsification: str, method: str) -> Hypothesis:
    return Hypothesis(identifier, statement, kind, outcome, null, falsification, method)  # type: ignore[arg-type]


def generate_seed_candidates(programme: ResearchProgramme) -> list[ResearchCandidate]:
    common = "Uses unchanged repeat, fixed transformation manifest, disjoint confirmatory sample, and task-cluster-aware uncertainty."
    candidates = [
        ResearchCandidate(
            "C1", "Cross-model permutation instability",
            "How does control-adjusted answer-order instability vary across model families and scales on NetOps MCQA tasks?",
            "Benchmark accuracy obscures item-level decision churn.",
            "paired semantic answer per question and model",
            ["model", "answer-order condition"],
            "absolute excess semantic disagreement",
            [_h("H1", "Permutation disagreement exceeds unchanged-repeat disagreement.", "primary", "absolute excess disagreement", "Excess is non-positive.", "Clustered interval includes zero and paired test is non-significant.", "Exact paired test plus task-cluster bootstrap")],
            ["at least three models", "disjoint sample"],
            "Cross-model evidence for benchmark instability.",
            ["effect absent across models"], common, 2700, 4.0,
            ["multi-model", "paired", "benchmark-validity"],
        ),
        ResearchCandidate(
            "C2", "Structured reasoning mitigation",
            "Can option-independent structured reasoning reduce answer-order instability without materially degrading accuracy?",
            "Diagnosis-only studies do not provide an operational mitigation.",
            "paired semantic answer per question and inference method",
            ["inference method", "answer-order condition"],
            "difference in excess disagreement between methods",
            [
                _h("H1", "Structured reasoning reduces excess disagreement versus direct MCQA.", "mitigation", "between-method difference in excess disagreement", "No reduction.", "Clustered interval includes zero or favours direct MCQA.", "Paired task-cluster bootstrap"),
                _h("H2", "Structured reasoning preserves accuracy within a non-inferiority margin.", "secondary", "accuracy difference", "Accuracy loss exceeds margin.", "Lower confidence bound crosses margin.", "Paired non-inferiority analysis"),
            ],
            ["direct baseline", "structured method", "repeat control"],
            "A tested inference-time mitigation for reliable NetOps decisions.",
            ["mitigation increases cost without reducing instability"],
            common + " Includes a preregistered accuracy non-inferiority margin.",
            1800, 5.0, ["mitigation", "paired", "operational"],
        ),
        ResearchCandidate(
            "C3", "Semantics-preserving transformation suite",
            "Which semantics-preserving presentation transformations most strongly alter LLM NetOps decisions?",
            "Answer permutation is only one form of irrelevant presentation change.",
            "paired semantic answer per question and transformation",
            ["transformation family"],
            "control-adjusted disagreement by transformation",
            [_h("H1", "At least one transformation yields positive excess disagreement.", "primary", "maximum preregistered excess disagreement", "All effects are non-positive.", "Multiplicity-adjusted intervals include zero for all transformations.", "Multiplicity-controlled paired analysis")],
            ["validated semantic equivalence", "identity controls"],
            "A taxonomy of benchmark-presentation fragility.",
            ["transformations cannot be validated as meaning-preserving"],
            common + " Includes identity controls and multiplicity correction.",
            4200, 8.0, ["transformation-suite", "benchmark-validity", "multi-condition"],
        ),
        ResearchCandidate(
            "C4", "Task-complexity moderation",
            "Do constraint density and competing objectives moderate transformation-induced NetOps decision instability?",
            "Mechanisms and task-level moderators remain unexplained.",
            "question with deterministic complexity features",
            ["constraint density", "objective conflict", "transformation"],
            "cluster-aware interaction coefficient",
            [_h("H1", "Higher deterministic complexity predicts greater excess disagreement.", "primary", "interaction coefficient", "Interaction is non-positive.", "Interval includes zero or coefficient is non-positive.", "Cluster-aware regression")],
            ["feature extraction validation", "adequate within-task spread"],
            "Mechanistic evidence on when instability occurs.",
            ["features lack reliability or variation"],
            common + " Uses preregistered deterministic feature extraction.",
            1200, 10.0, ["mechanism", "moderation", "paired"],
        ),
        ResearchCandidate(
            "C5", "Permutation ensemble mitigation",
            "Can semantic voting over fixed option permutations improve decision stability and calibration?",
            "Permutation ensembles are plausible but their cost-benefit is unclear.",
            "question-level semantic ensemble decision",
            ["ensemble size", "aggregation rule"],
            "excess disagreement and cost-adjusted stability",
            [_h("H1", "Permutation voting reduces excess disagreement versus one-pass answering.", "mitigation", "difference in excess disagreement", "Voting provides no reduction.", "Clustered interval includes zero.", "Paired task-cluster bootstrap")],
            ["fixed permutation set", "semantic aggregation"],
            "A robustness wrapper with explicit cost trade-offs.",
            ["cost increase dominates robustness gain"],
            common + " Uses preregistered semantic voting.",
            6000, 7.0, ["mitigation", "ensemble", "cost"],
        ),
    ]
    result = candidates[:programme.candidate_count]
    for candidate in result:
        candidate.validate()
    return result
