from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from agents import Agent, Runner

from .repair_schemas import (
    RepairedStudyDesign,
    RepairReadinessReport,
)


T = TypeVar("T")


DESIGN_REPAIR_AGENT = Agent(
    name="Autonomous Study Design Repairer",
    model="gpt-5-mini",
    output_type=RepairedStudyDesign,
    instructions=(
        "You repair the selected study design before preregistration. "
        "Use only the supplied evidence, candidate, decision, capability "
        "manifest, and required repairs. "

        "The repaired design must not depend on unavailable resources, "
        "including human labour, human annotation, manual adjudication, "
        "external partners, private infrastructure, or local GPU "
        "execution unless explicitly available in the capability "
        "manifest. "

        "When a procedure is automated, avoid ambiguous phrases such as "
        "'manual review', 'human review', or 'manual adjudication'. "
        "Describe it instead as deterministic flagging, automatic "
        "exclusion, automated re-execution, or sensitivity analysis. "
        "Do not introduce any required human annotation, adjudication, "
        "scoring, evaluation, or scientific review. "

        "Cross-check every sample count across sampling_plan, power_plan, "
        "budget_scenarios, and any execution estimates. Per-cluster, "
        "per-method, per-condition, and total counts must be "
        "arithmetically consistent. Explicitly state how the totals are "
        "derived, for example: number of methods multiplied by trials per "
        "method equals the overall confirmatory trial count. "
        
        "Repair the selected study using only the supplied programme, "
        "capability manifest, evidence, reviews, evidence verification, "
        "and required repairs. Do not preserve development choices by "
        "default. Preserve the selected candidate identity unless the "
        "input explicitly requires rejection rather than repair. "
        "\n\n"
        "Treat the frozen capability manifest as a hard boundary. The "
        "repaired study must be scientifically meaningful and executable "
        "using only the listed resources. Do not require unavailable "
        "human labour, human annotation, manual adjudication, external "
        "partners, NDA resources, private laboratories, Kubernetes, or "
        "local GPU execution. Use hosted APIs or CPU-compatible methods "
        "when appropriate. "
        "\n\n"
        "The input includes available_analysis_families and "
        "available_analysis_contracts. Treat these as a hard scientific "
        "execution boundary. The repaired design's confirmatory primary "
        "estimand and analysis plan must be computable by at least one "
        "supplied registered analysis executor. Do not design or retain "
        "a primary estimand merely because it is scientifically desirable "
        "when no supplied analysis contract can compute it. You may "
        "autonomously reformulate the selected candidate into a meaningful "
        "study that fits the available analysis machinery, while preserving "
        "the selected candidate identity and broad research question. "
        "Do not invent an analysis executor or unsupported estimand. "
        "\n\n"
        "Resolve contamination, budget, power, transformation validation, "
        "model scope, multiplicity, missingness, and confirmatory versus "
        "exploratory claims. Provide at least two budget scenarios and "
        "identify one recommended scenario. Use only supplied evidence IDs. "
        "\n\n"
        "Classify issues carefully. Put an issue in "
        "unresolved_critical_issues only when it prevents the proposed "
        "study from testing its hypotheses, identifying its estimands, "
        "executing within the frozen capabilities, producing interpretable "
        "results, or preserving preregistration integrity. "
        "\n\n"
        "Do not classify ordinary scope limitations, residual uncertainty, "
        "external-validity limits, commercial-API nondeterminism, residual "
        "detector error, or inability to generalize beyond the declared "
        "benchmark population as critical when the claims are explicitly "
        "bounded to the available public or synthetic data. Place those in "
        "remaining_noncritical_uncertainties and narrow the claims "
        "accordingly. "
        "\n\n"
        "For example, a public or synthetic NetOps benchmark need not prove "
        "generalization to private production operator networks. That is a "
        "noncritical external-validity limitation when the research question "
        "and conclusions are explicitly limited to the benchmark scope. It "
        "becomes critical only if the confirmatory hypothesis itself claims "
        "production-network generalization without evidence capable of "
        "testing that claim. "
        "\n\n"
        "Before returning, verify that preregistration_fields_complete is "
        "true only when the research question, hypotheses, estimands, "
        "sampling, analysis, multiplicity, missingness, contamination, "
        "budget, power, and transformation-validation plans are all "
        "complete. Every remaining critical issue must be concrete, "
        "unresolved, and genuinely blocking."
    ),
)


READINESS_JUDGE = Agent(
    name="Preregistration Readiness Judge",
    model="gpt-5-mini",
    output_type=RepairReadinessReport,
    instructions=(
        "Judge whether the repaired design is ready to proceed toward a "
        "future final autonomous run. Evaluate only the supplied repaired "
        "design, evidence verification, capability manifest, and selected "
        "candidate identity. "
        "\n\n"
        "Fail readiness for missing or unresolved evidence, incomplete "
        "bibliographic identity, candidate mismatch, absent budget or power "
        "plans, absent transformation validation, incomplete sampling or "
        "analysis plans, missing multiplicity or missingness treatment, "
        "missing contamination controls, infeasibility under the frozen "
        "capability manifest, or genuinely unresolved critical issues. "
        "\n\n"
        "A critical issue is one that prevents hypothesis testing, valid "
        "estimation, executable autonomous implementation, interpretable "
        "results, or preregistration integrity. "
        "\n\n"
        "Do not fail readiness merely because the study cannot establish "
        "external validity beyond its declared population. Public-data-only "
        "or synthetic-benchmark studies may proceed when their research "
        "question, hypotheses, estimands, and conclusions are explicitly "
        "bounded to that scope. Treat ecological generalization to private "
        "production operator networks as a noncritical limitation unless "
        "the study makes a confirmatory production-generalization claim. "
        "\n\n"
        "Residual API nondeterminism, residual sanitization risk, incomplete "
        "real-world representativeness, and future governance needs should "
        "normally be warnings rather than failed gates when they do not "
        "invalidate the declared confirmatory analysis. "
        "\n\n"
        "When all blocking requirements are satisfied, report readiness and "
        "leave failed_gates empty. The successful state describes design "
        "readiness only; it must never claim that execution, analysis, or "
        "final submission has already occurred."
    ),
)


async def run_agent_with_retry(
    agent: Agent,
    payload: dict[str, Any],
    *,
    expected_type: type[T],
    stage_name: str,
    attempts: int = 3,
) -> T:
    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            result = await Runner.run(
                agent,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            )

            output = result.final_output

            if not isinstance(
                output,
                expected_type,
            ):
                raise TypeError(
                    f"{stage_name}: expected "
                    f"{expected_type.__name__}, received "
                    f"{type(output).__name__}"
                )

            return output

        except Exception as exc:
            last_error = exc

            if attempt == attempts:
                break

            delay_seconds = (
                5 * (2 ** (attempt - 1))
            )

            print(
                f"{stage_name} failed "
                f"{attempt}/{attempts}: {exc}"
            )

            await asyncio.sleep(
                delay_seconds
            )

    raise RuntimeError(
        f"{stage_name} failed after "
        f"{attempts} attempts"
    ) from last_error