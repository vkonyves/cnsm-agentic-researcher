from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from agents import Agent, Runner

from .design_repair import (
    DESIGN_REPAIR_AGENT,
    READINESS_JUDGE,
    run_agent_with_retry,
)
from .evidence_verification import (
    verify_evidence,
)
from .execution_adapters import (
    resolve_adapter,
)
from .feasibility import (
    feasibility_report,
)
from .final_agents import (
    ANALYSIS_PLANNER,
    EXPERIMENT_PLANNER,
    FINAL_JUDGE,
    MANUSCRIPT_AUTHOR,
    MANUSCRIPT_REVISER,
    PEER_REVIEWER,
    PREREGISTRATION_AGENT,
)
from .final_guardrails import (
    assert_fresh_run_dir,
    assert_no_development_inputs,
    sha256_file,
)
from .final_schemas import (
    AnalysisPlan,
    ExperimentPlan,
    FinalReadinessReport,
    ManuscriptPackage,
    PeerReviewReport,
    PreregistrationDocument,
)
from .pipeline import (
    AutonomousDiscoveryPipeline,
)
from .repair_schemas import (
    RepairedStudyDesign,
    RepairReadinessReport,
)


T = TypeVar("T")


def read_json(
    path: Path,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(value, "model_dump"):
        value = value.model_dump()

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_state(
    *,
    run_dir: Path,
    state: str,
    selected_candidate_id: str | None,
    development_rehearsal: bool,
    additional_fields: dict[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "state": state,
        "selected_candidate_id": (
            selected_candidate_id
        ),
        "development_rehearsal": (
            development_rehearsal
        ),
        "updated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    if additional_fields:
        value.update(
            additional_fields
        )

    write_json(
        run_dir / "state.json",
        value,
    )


async def run_agent(
    agent: Agent,
    payload: dict[str, Any],
    *,
    expected_type: type[T],
    stage_name: str,
    attempts: int = 3,
) -> T:
    """
    Run a structured-output agent with bounded retries.

    Development-run paths are prohibited from entering
    final scientific-stage prompts.
    """
    assert_no_development_inputs(
        payload
    )

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
                    f"{stage_name} returned "
                    f"{type(output).__name__}; "
                    f"expected "
                    f"{expected_type.__name__}."
                )

            return output

        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            delay_seconds = (
                5 * (2 ** (attempt - 1))
            )

            print(
                f"{stage_name} failed on attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            print(
                "Retrying in "
                f"{delay_seconds} seconds..."
            )

            await asyncio.sleep(
                delay_seconds
            )

    raise RuntimeError(
        f"{stage_name} failed after "
        f"{attempts} attempts."
    ) from last_error


def create_failure_report(
    *,
    passed_gates: list[str],
    failed_gate: str,
    final_state: str,
    warnings: list[str] | None = None,
) -> FinalReadinessReport:
    return FinalReadinessReport(
        ready=False,
        passed_gates=passed_gates,
        failed_gates=[
            failed_gate
        ],
        warnings=warnings or [],
        final_state=final_state,
    )


class FinalAutonomousResearchPipeline:
    def __init__(
        self,
        *,
        model: str,
        development_rehearsal: bool,
    ) -> None:
        self.model = model
        self.development_rehearsal = (
            development_rehearsal
        )

        for agent in (
            DESIGN_REPAIR_AGENT,
            READINESS_JUDGE,
            PREREGISTRATION_AGENT,
            EXPERIMENT_PLANNER,
            ANALYSIS_PLANNER,
            MANUSCRIPT_AUTHOR,
            PEER_REVIEWER,
            MANUSCRIPT_REVISER,
            FINAL_JUDGE,
        ):
            agent.model = model

    async def run(
        self,
        *,
        master_prompt: str,
        run_dir: Path,
        capability_manifest: dict[str, Any],
    ) -> FinalReadinessReport:
        assert_fresh_run_dir(
            run_dir,
            development_rehearsal=(
                self.development_rehearsal
            ),
        )

        assert_no_development_inputs(
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
            }
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for relative_directory in (
            "literature",
            "selection",
            "design",
            "preregistration",
            "execution",
            "analysis",
            "analysis/tables",
            "analysis/figures",
            "manuscript",
            "manuscript/review_rounds",
            "disclosure",
        ):
            (
                run_dir
                / relative_directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        programme = {
            "programme_name": (
                "CNSM 2026 final autonomous run"
            ),
            "master_prompt": master_prompt,
            "topic_family": (
                "Generative AI and Large "
                "Language Models for NetOps"
            ),
            "capability_manifest": (
                capability_manifest
            ),
            "development_rehearsal": (
                self.development_rehearsal
            ),
        }

        # -------------------------------------------------
        # 1. Fresh autonomous discovery
        # -------------------------------------------------

        discovery_pipeline = (
            AutonomousDiscoveryPipeline(
                model=self.model,
                per_source_per_query=8,
                max_synthesis_records=80,
            )
        )

        decision = (
            await discovery_pipeline.run(
                programme=programme,
                run_dir=run_dir,
            )
        )

        selected_candidate_id = (
            decision.selected_candidate_id
        )

        if not selected_candidate_id:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                ],
                failed_gate=(
                    "Autonomous discovery completed without "
                    "a final selected candidate."
                ),
                final_state=(
                    "AUTONOMOUS_FINALIST_RESOLUTION_REQUIRED"
                ),
                warnings=[
                    (
                        "The discovery stage returned no "
                        "selected_candidate_id. The run was "
                        "stopped before preregistration."
                    )
                ],
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=None,
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "discovery_decision": (
                        decision.model_dump()
                        if hasattr(
                            decision,
                            "model_dump",
                        )
                        else str(decision)
                    ),
                },
            )

            return report

        literature_dir = (
            run_dir / "literature"
        )
        selection_dir = (
            run_dir / "selection"
        )
        design_dir = (
            run_dir / "design"
        )

        records = read_json(
            literature_dir
            / "records.json"
        )
        synthesis = read_json(
            literature_dir
            / "evidence_synthesis.json"
        )
        candidates = read_json(
            selection_dir
            / "candidates.json"
        )
        reviews = read_json(
            selection_dir
            / "critic_reviews.json"
        )
        decision_json = read_json(
            selection_dir
            / "decision.json"
        )
        candidate_validation = read_json(
            selection_dir
            / "candidate_validation.json"
        )

        if (
            candidate_validation.get(
                "candidate_validation_status"
            )
            != "passed"
        ):
            raise ValueError(
                "Candidate validation did not pass."
            )

        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate
            in candidates["candidates"]
        }

        if (
            selected_candidate_id
            not in candidates_by_id
        ):
            raise ValueError(
                "Selected candidate does not exist "
                "in the validated candidate set."
            )

        selected_candidate = (
            candidates_by_id[
                selected_candidate_id
            ]
        )

        # -------------------------------------------------
        # 2. Independent evidence verification
        # -------------------------------------------------

        evidence_verification = (
            verify_evidence(
                records=records,
                synthesis=synthesis,
                candidates=candidates,
                decision=decision_json,
            )
        )

        evidence_report = (
            evidence_verification.to_dict()
        )

        write_json(
            design_dir
            / "evidence_verification.json",
            evidence_report,
        )

        if (
            evidence_verification
            .critical_issues
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                ],
                failed_gate=(
                    "Evidence verification found "
                    "critical issues."
                ),
                final_state=(
                    "AUTONOMOUS_EVIDENCE_REPAIR_REQUIRED"
                ),
                warnings=(
                    evidence_verification
                    .warnings
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "evidence_critical_issues": (
                        evidence_verification
                        .critical_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 3. Autonomous design repair
        # -------------------------------------------------

        allowed_evidence_record_ids = sorted(
            {
                record["record_id"]
                for record in records
            }
        )

        repair_payload = {
            "programme": programme,
            "master_prompt": master_prompt,
            "capability_manifest": (
                capability_manifest
            ),
            "selected_candidate": (
                selected_candidate
            ),
            "selection_decision": (
                decision_json
            ),
            "critic_reviews": reviews,
            "evidence_synthesis": synthesis,
            "evidence_verification": (
                evidence_report
            ),
            "allowed_evidence_record_ids": (
                allowed_evidence_record_ids
            ),
        }

        repaired_design = (
            await run_agent_with_retry(
                DESIGN_REPAIR_AGENT,
                repair_payload,
                expected_type=(
                    RepairedStudyDesign
                ),
                stage_name=(
                    "Autonomous design repair"
                ),
            )
        )

        if (
            repaired_design
            .selected_candidate_id
            != selected_candidate_id
        ):
            raise ValueError(
                "Autonomous design repair changed "
                "the selected candidate ID."
            )

        unknown_repair_evidence_ids = sorted(
            set(
                repaired_design
                .evidence_record_ids
            )
            - set(
                allowed_evidence_record_ids
            )
        )

        if unknown_repair_evidence_ids:
            raise ValueError(
                "Repaired design references unknown "
                "evidence IDs: "
                f"{unknown_repair_evidence_ids}"
            )

        write_json(
            design_dir
            / "repaired_design.json",
            repaired_design,
        )

        repair_readiness = (
            await run_agent_with_retry(
                READINESS_JUDGE,
                {
                    "programme": programme,
                    "master_prompt": (
                        master_prompt
                    ),
                    "capability_manifest": (
                        capability_manifest
                    ),
                    "selected_candidate_id": (
                        selected_candidate_id
                    ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "repaired_design": (
                        repaired_design
                        .model_dump()
                    ),
                },
                expected_type=(
                    RepairReadinessReport
                ),
                stage_name=(
                    "Design-repair readiness judgement"
                ),
            )
        )

        if (
            repair_readiness
            .selected_candidate_id
            != selected_candidate_id
        ):
            raise ValueError(
                "Design-repair readiness report "
                "candidate mismatch."
            )

        repair_ready = (
            not (
                evidence_verification
                .critical_issues
            )
            and (
                repaired_design
                .preregistration_fields_complete
            )
            and not (
                repaired_design
                .unresolved_critical_issues
            )
        )

        repair_readiness.next_state = (
            "DESIGN_REPAIRED"
            if repair_ready
            else "DESIGN_REPAIR_REQUIRED"
        )

        write_json(
            design_dir
            / "repair_readiness_report.json",
            repair_readiness,
        )

        if not repair_ready:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                ],
                failed_gate=(
                    "Autonomous design repair left "
                    "critical issues unresolved."
                ),
                final_state=(
                    "AUTONOMOUS_DESIGN_REPAIR_REQUIRED"
                ),
                warnings=(
                    repaired_design
                    .remaining_noncritical_uncertainties
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "unresolved_critical_issues": (
                        repaired_design
                        .unresolved_critical_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 4. Deterministic feasibility of repaired design
        # -------------------------------------------------

        repaired_design_feasibility = (
            feasibility_report(
                design=(
                    repaired_design
                    .model_dump()
                ),
                capability_manifest=(
                    capability_manifest
                ),
            )
        )

        write_json(
            design_dir
            / "repaired_design_feasibility.json",
            repaired_design_feasibility,
        )

        if (
            repaired_design_feasibility[
                "status"
            ]
            != "passed"
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                ],
                failed_gate=(
                    "Repaired design is infeasible "
                    "under the frozen capability manifest."
                ),
                final_state=(
                    "AUTONOMOUS_DESIGN_REPAIR_REQUIRED"
                ),
                warnings=(
                    repaired_design_feasibility[
                        "issues"
                    ]
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "feasibility_issues": (
                        repaired_design_feasibility[
                            "issues"
                        ]
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 5. Autonomous preregistration
        # -------------------------------------------------

        preregistration = await run_agent(
            PREREGISTRATION_AGENT,
            {
                "master_prompt": (
                    master_prompt
                ),
                "capability_manifest": (
                    capability_manifest
                ),
                "selected_candidate_id": (
                    selected_candidate_id
                ),
                "repaired_design": (
                    repaired_design
                    .model_dump()
                ),
                "evidence_verification": (
                    evidence_report
                ),
                "verified_record_ids": (
                    allowed_evidence_record_ids
                ),
            },
            expected_type=(
                PreregistrationDocument
            ),
            stage_name="Preregistration",
        )

        preregistration_path = (
            run_dir
            / "preregistration"
            / "preregistration.json"
        )

        write_json(
            preregistration_path,
            preregistration,
        )

        preregistration_hash = (
            sha256_file(
                preregistration_path
            )
        )

        (
            preregistration_path.parent
            / "preregistration.sha256"
        ).write_text(
            preregistration_hash + "\n",
            encoding="utf-8",
        )

        # -------------------------------------------------
        # 6. Autonomous experiment planning
        # -------------------------------------------------

        experiment_plan = await run_agent(
            EXPERIMENT_PLANNER,
            {
                "master_prompt": (
                    master_prompt
                ),
                "capability_manifest": (
                    capability_manifest
                ),
                "preregistration": (
                    preregistration
                    .model_dump()
                ),
                "repaired_design": (
                    repaired_design
                    .model_dump()
                ),
                "verified_records": records,
            },
            expected_type=ExperimentPlan,
            stage_name=(
                "Experiment planning"
            ),
        )

        experiment_plan_path = (
            run_dir
            / "execution"
            / "experiment_plan.json"
        )

        write_json(
            experiment_plan_path,
            experiment_plan,
        )

        # -------------------------------------------------
        # 7. Deterministic experiment-plan feasibility
        # -------------------------------------------------

        experiment_feasibility = (
            feasibility_report(
                design=(
                    experiment_plan
                    .model_dump()
                ),
                capability_manifest=(
                    capability_manifest
                ),
            )
        )

        write_json(
            design_dir
            / "experiment_plan_feasibility.json",
            experiment_feasibility,
        )

        if (
            experiment_feasibility[
                "status"
            ]
            != "passed"
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "sealed_preregistration",
                    "experiment_plan",
                ],
                failed_gate=(
                    "Experiment plan is infeasible "
                    "under the frozen capability manifest."
                ),
                final_state=(
                    "AUTONOMOUS_DESIGN_REPAIR_REQUIRED"
                ),
                warnings=(
                    experiment_feasibility[
                        "issues"
                    ]
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "feasibility_issues": (
                        experiment_feasibility[
                            "issues"
                        ]
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 8. Resolve autonomous execution adapter
        # -------------------------------------------------

        adapter = resolve_adapter(
            experiment_plan.model_dump()
        )

        if adapter is None:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "sealed_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                ],
                failed_gate=(
                    "No installed autonomous execution "
                    "adapter supports the repaired study."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_ADAPTER_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
            )

            return report

        # -------------------------------------------------
        # 9. Autonomous experiment execution
        # -------------------------------------------------

        execution_manifest = adapter.execute(
            plan=experiment_plan.model_dump(),
            preregistration=(
                preregistration.model_dump()
            ),
            output_dir=(
                run_dir / "execution"
            ),
        )

        write_json(
            run_dir
            / "execution"
            / "execution_manifest.json",
            execution_manifest,
        )

        if (
            execution_manifest.get(
                "status"
            )
            != "COMPLETED"
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "sealed_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "execution_adapter_resolved",
                ],
                failed_gate=(
                    "Autonomous execution adapter "
                    "did not complete successfully."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_INCOMPLETE"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
            )

            return report

        # -------------------------------------------------
        # 10. Analysis planning and execution check
        # -------------------------------------------------

        analysis_plan = await run_agent(
            ANALYSIS_PLANNER,
            {
                "master_prompt": (
                    master_prompt
                ),
                "capability_manifest": (
                    capability_manifest
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "experiment_plan": (
                    experiment_plan.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
            },
            expected_type=AnalysisPlan,
            stage_name="Analysis planning",
        )

        write_json(
            run_dir
            / "analysis"
            / "analysis_plan.json",
            analysis_plan,
        )

        results_path = (
            run_dir
            / "analysis"
            / "results.json"
        )

        if not results_path.is_file():
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                    "analysis_plan",
                ],
                failed_gate=(
                    "Autonomous analysis executor has "
                    "not produced results.json."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_EXECUTOR_REQUIRED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                report,
            )

            write_state(
                run_dir=run_dir,
                state=report.final_state,
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
            )

            return report

        analysis_results = read_json(
            results_path
        )

        # -------------------------------------------------
        # 11. Autonomous manuscript generation
        # -------------------------------------------------

        draft = await run_agent(
            MANUSCRIPT_AUTHOR,
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "verified_records": records,
                "evidence_verification": (
                    evidence_report
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_plan": (
                    analysis_plan.model_dump()
                ),
                "analysis_results": (
                    analysis_results
                ),
            },
            expected_type=ManuscriptPackage,
            stage_name=(
                "Manuscript drafting"
            ),
        )

        write_json(
            run_dir
            / "manuscript"
            / "draft_package.json",
            draft,
        )

        # -------------------------------------------------
        # 12. AI peer review
        # -------------------------------------------------

        peer_review = await run_agent(
            PEER_REVIEWER,
            {
                "master_prompt": master_prompt,
                "evidence_verification": (
                    evidence_report
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_results": (
                    analysis_results
                ),
                "manuscript": (
                    draft.model_dump()
                ),
            },
            expected_type=PeerReviewReport,
            stage_name="AI peer review",
        )

        write_json(
            run_dir
            / "manuscript"
            / "review_rounds"
            / "review_01.json",
            peer_review,
        )

        # -------------------------------------------------
        # 13. Autonomous revision
        # -------------------------------------------------

        revised_manuscript = await run_agent(
            MANUSCRIPT_REVISER,
            {
                "master_prompt": master_prompt,
                "verified_records": records,
                "evidence_verification": (
                    evidence_report
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_results": (
                    analysis_results
                ),
                "manuscript": (
                    draft.model_dump()
                ),
                "peer_review": (
                    peer_review.model_dump()
                ),
            },
            expected_type=ManuscriptPackage,
            stage_name=(
                "Autonomous manuscript revision"
            ),
        )

        write_json(
            run_dir
            / "manuscript"
            / "revised_package.json",
            revised_manuscript,
        )

        # -------------------------------------------------
        # 14. Final autonomous readiness judgement
        # -------------------------------------------------

        final_report = await run_agent(
            FINAL_JUDGE,
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "evidence_verification": (
                    evidence_report
                ),
                "repaired_design": (
                    repaired_design.model_dump()
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    execution_manifest
                ),
                "analysis_results": (
                    analysis_results
                ),
                "manuscript": (
                    revised_manuscript
                    .model_dump()
                ),
                "peer_review": (
                    peer_review.model_dump()
                ),
            },
            expected_type=(
                FinalReadinessReport
            ),
            stage_name=(
                "Final readiness judgement"
            ),
        )

        write_json(
            run_dir
            / "final_readiness_report.json",
            final_report,
        )

        write_state(
            run_dir=run_dir,
            state=final_report.final_state,
            selected_candidate_id=(
                selected_candidate_id
            ),
            development_rehearsal=(
                self.development_rehearsal
            ),
            additional_fields={
                "ready": final_report.ready,
            },
        )

        return final_report