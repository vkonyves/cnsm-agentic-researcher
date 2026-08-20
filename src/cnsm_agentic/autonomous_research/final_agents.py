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
        "hypotheses, estimands, or analysis commitments. "
        "\n\n"
        "The input contains available_adapter_families. Set "
        "adapter_family to exactly one identifier from "
        "available_adapter_families. Do not invent, describe, expand, "
        "rename, decorate, combine, or qualify the identifier. The "
        "adapter_family field must contain only the exact registered "
        "identifier. "
        "\n\n"
        "The input also contains available_adapter_contracts. The "
        "selected adapter's contract is authoritative for the "
        "machine-readable execution fields of ExperimentPlan. Populate "
        "execution_mode, design, conditions, task_families, "
        "transformations, result_schema_id, result_schema_version, "
        "model_provider, model_name, model_version, "
        "deterministic_automated_scoring, "
        "requires_human_scientific_labour, task_count, task_indices, "
        "estimated_model_calls, maximum_model_calls, reasoning_effort, "
        "maximum_attempts_per_call, and max_output_tokens explicitly. "
        "Do not merely describe these requirements in prose fields. "
        "\n\n"
        "Do not claim that a study is autonomously executable when "
        "available_adapter_families is empty. Do not substitute a "
        "general technology family, hosted-API description, software "
        "stack, implementation concept, or proposed future adapter "
        "for a registered adapter identifier. "
        "\n\n"
        "The implementation strategy, public resources, model plan, "
        "task manifest, transformation manifest, execution batches, "
        "randomisation, caching, failure recovery, result schema, "
        "model-call estimate, and compute notes must all fit the real "
        "scope of the selected registered adapter. Do not request "
        "capabilities merely because they would be scientifically "
        "useful. "
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
        "\n- Docker, Mininet, Batfish, FAISS, network emulation, "
        "formal verification, custom DSL execution, or external "
        "simulation unless both the capability manifest and the "
        "selected registered adapter explicitly support them;"
        "\n- model calls above maximum_planned_model_calls."
        "\n\n"
        "Do not retain unavailable resources as optional, "
        "recommended, fallback, audit, validation, sensitivity, or "
        "future components. Remove them entirely from the executable "
        "plan. When no local GPU is available, use hosted model APIs "
        "or CPU-compatible methods only when they are supported by "
        "the selected adapter. "
        "\n\n"
        "When autonomous scoring is required, all labels, audits, "
        "validations, scoring, and adjudication must be performed by "
        "deterministic code or by predeclared autonomous scorers "
        "implemented by the selected adapter. Avoid ambiguous wording "
        "such as 'manual review', 'human review', or 'manual "
        "adjudication' when the procedure is automated. Describe "
        "deterministic flagging, automatic exclusion, automated "
        "re-execution, or sensitivity analysis instead. "
        "\n\n"
        "Cross-check every numerical quantity across the "
        "preregistration, sampling plan, execution batches, estimated "
        "model calls, and compute notes. Per-cluster, per-method, "
        "per-condition, per-batch, and total counts must be "
        "arithmetically consistent. Explicitly state how totals are "
        "derived. "
        "\n\n"
        "If a rejected previous plan and deterministic feasibility "
        "issues are supplied, repair every listed issue explicitly. "
        "Replace each forbidden dependency rather than renaming it, "
        "softening it, or describing it as optional. Preserve the "
        "scientific question and estimands only within the actual "
        "capabilities of a registered adapter."
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
        "analysis, and planned tables and figures. "
        "\n\n"
        "The input contains available_analysis_families. Set "
        "analysis_executor to exactly one identifier from "
        "available_analysis_families. Do not invent, describe, expand, "
        "rename, decorate, combine, or qualify the identifier. The "
        "analysis_executor field must contain only the exact "
        "registered identifier. "
        "\n\n"
        "Do not claim that analysis is autonomously executable when "
        "available_analysis_families is empty. Do not substitute a "
        "statistical method name, software library, prose "
        "description, or proposed future executor for a registered "
        "analysis-executor identifier. "
        "\n\n"
        "The proposed primary, secondary, sensitivity, uncertainty, "
        "multiplicity, contamination, and failed-call analyses must "
        "be executable by the selected deterministic analysis "
        "executor. Use only fields actually provided by the completed "
        "execution manifest and its referenced raw-result artifacts. "
        "Do not assume unavailable variables, labels, annotations, "
        "metrics, logs, or metadata. "
        "\n\n"
        "Do not introduce human review, human adjudication, manual "
        "scoring, external statistical analysis, unregistered "
        "software services, or new model calls. Do not alter the "
        "sealed confirmatory estimand after seeing the execution "
        "results. "
        "\n\n"
        "Cross-check all analysis denominators, sample sizes, paired "
        "units, strata, exclusions, failed calls, and multiplicity "
        "families against the execution manifest and sealed "
        "preregistration. Every table and figure specification must "
        "be producible by the selected executor from existing "
        "artifacts."
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