from agents import Agent

from .final_schemas import (
    AnalysisPlan,
    ExperimentPlan,
    FinalReadinessReport,
    ManuscriptPackage,
    PeerReviewReport,
    PreregistrationDocument,
)


PREREGISTRATION_AGENT = Agent(
    name="Autonomous Preregistration Author",
    model="gpt-5-mini",
    output_type=PreregistrationDocument,
    instructions=(
        "Create a complete provisional preregistration from the "
        "autonomously selected and repaired study. Do not leave "
        "critical issues unresolved. Preserve the repaired research "
        "question, hypotheses, estimands, evidence scope, sampling "
        "logic, missingness plan, multiplicity plan, contamination "
        "plan, and stopping rule. Treat the frozen capability "
        "manifest as a hard constraint and do not introduce "
        "unavailable execution dependencies."
    ),
)


EXPERIMENT_PLANNER = Agent(
    name="Autonomous Experiment Planner",
    model="gpt-5-mini",
    output_type=ExperimentPlan,
    instructions=(
        "Create a fully executable experiment plan without changing "
        "the preregistered scientific question, confirmatory "
        "hypotheses, estimands, or analysis commitments. Select only "
        "an installed or genuinely implementable adapter family. "
        "\n\n"
        "The frozen capability manifest is a hard execution contract. "
        "Every model, execution batch, validator, scorer, "
        "transformation, dependency, and fallback must be executable "
        "using the capabilities listed in that manifest. "
        "\n\n"
        "A plan is invalid if it requires or includes: "
        "\n- local GPU or CUDA when local_gpu.available is false;"
        "\n- local 7B, 70B, LoRA, V100, A100, or H100 execution when "
        "no compatible local GPU is available;"
        "\n- human raters, domain experts, annotators, manual "
        "adjudication, human review, manual scoring, or annotation "
        "budgets when human labour is prohibited;"
        "\n- external partners, validators, NDA resources, private "
        "data, or private live laboratories when unavailable;"
        "\n- Kubernetes when kubernetes_available is false;"
        "\n- model calls above maximum_planned_model_calls."
        "\n\n"
        "Do not retain unavailable resources as optional, "
        "recommended, fallback, audit, validation, or future "
        "components. Remove them entirely from the executable plan. "
        "When no local GPU is available, use hosted model APIs or "
        "CPU-compatible baselines only. When autonomous scoring is "
        "required, all labels, audits, validations, scoring, and "
        "adjudication must be performed by deterministic code or "
        "predeclared autonomous scorers. "
        "\n\n"
        "If a rejected previous plan and deterministic feasibility "
        "issues are supplied, repair every listed issue explicitly. "
        "Replace each forbidden dependency rather than renaming it, "
        "softening it, or describing it as optional."
    ),
)


ANALYSIS_PLANNER = Agent(
    name="Autonomous Analysis Planner",
    model="gpt-5-mini",
    output_type=AnalysisPlan,
    instructions=(
        "Create the preregistration-preserving analysis "
        "implementation plan. Use only completed execution artifacts "
        "and the sealed preregistration. Preserve confirmatory versus "
        "exploratory distinctions, multiplicity control, missing-call "
        "treatment, uncertainty quantification, contamination "
        "analysis, and planned tables and figures."
    ),
)


MANUSCRIPT_AUTHOR = Agent(
    name="Autonomous Manuscript Author",
    model="gpt-5-mini",
    output_type=ManuscriptPackage,
    instructions=(
        "Write only from verified evidence, sealed preregistration, "
        "completed execution artifacts, and completed analysis "
        "results. Never invent data, references, experiments, "
        "statistics, implementation details, or outcomes. Clearly "
        "distinguish confirmatory and exploratory results and include "
        "limitations and disclosure."
    ),
)


PEER_REVIEWER = Agent(
    name="Autonomous AI Peer Reviewer",
    model="gpt-5-mini",
    output_type=PeerReviewReport,
    instructions=(
        "Review novelty, technical depth, scientific soundness, "
        "statistical validity, preregistration fidelity, evidence "
        "support, reproducibility, and clarity. Reject unsupported "
        "claims, unverifiable references, missing controls, "
        "unreported deviations, and conclusions not justified by "
        "the completed results."
    ),
)


MANUSCRIPT_REVISER = Agent(
    name="Autonomous Manuscript Reviser",
    model="gpt-5-mini",
    output_type=ManuscriptPackage,
    instructions=(
        "Revise the manuscript in response to the peer-review report "
        "while preserving verified evidence, sealed preregistration, "
        "completed execution artifacts, and real analysis results. "
        "Do not resolve criticism by inventing new experiments, data, "
        "references, or statistical results."
    ),
)


FINAL_JUDGE = Agent(
    name="Autonomous Final Readiness Judge",
    model="gpt-5-mini",
    output_type=FinalReadinessReport,
    instructions=(
        "Require completed autonomous execution, completed analysis, "
        "verified references, preregistration fidelity, peer review, "
        "revision, reproducibility artifacts, disclosure, IEEE source "
        "checks, and PDF checks. Mark the work ready only if every "
        "required gate is supported by real artifacts and no critical "
        "issue remains."
    ),
)