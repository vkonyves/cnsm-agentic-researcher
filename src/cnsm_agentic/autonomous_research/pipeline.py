from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, TypeVar

from agents import Agent, Runner

from .agents import (
    CANDIDATE_CRITIC,
    CANDIDATE_GENERATOR,
    EVIDENCE_SYNTHESISER,
    QUERY_PLANNER,
    SELECTION_JUDGE,
)
from .evidence_verification import normalise_evidence_id
from .literature import discover_literature
from .schemas import (
    CandidateSet,
    EvidenceSynthesis,
    LiteratureRecord,
    QueryPlan,
    ReviewSet,
    SelectionDecision,
)


T = TypeVar("T")

MAX_SYNTHESIS_RECORDS = 80
MAX_ABSTRACT_CHARACTERS = 1_500
MAX_AGENT_ATTEMPTS = 3


def _write_json(
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


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


async def _run_agent_with_retry(
    agent: Agent,
    payload: dict[str, Any],
    *,
    expected_type: type[T],
    stage_name: str,
    attempts: int = MAX_AGENT_ATTEMPTS,
) -> T:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = await Runner.run(
                agent,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            )

            output = result.final_output

            if not isinstance(output, expected_type):
                raise TypeError(
                    f"{stage_name} returned "
                    f"{type(output).__name__}, expected "
                    f"{expected_type.__name__}"
                )

            return output

        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            delay_seconds = 5 * (2 ** (attempt - 1))

            print(
                f"{stage_name} failed on attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            print(
                f"Retrying in {delay_seconds} seconds..."
            )

            await asyncio.sleep(delay_seconds)

    raise RuntimeError(
        f"{stage_name} failed after "
        f"{attempts} attempts"
    ) from last_error


async def _generate_candidate_set_with_repair(
    *,
    payload: dict[str, Any],
    attempts_dir: Path,
    attempts: int = MAX_AGENT_ATTEMPTS,
) -> CandidateSet:
    """
    Generate a complete CandidateSet with bounded structural repair.

    Scientific candidate content remains autonomous. On a rejected model
    output, only the deterministic validation/serialization failure is
    supplied to the next attempt.
    """
    attempts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_error: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        attempt_payload = dict(payload)

        if previous_error is not None:
            attempt_payload[
                "candidate_generation_repair"
            ] = {
                "previous_output_rejected": True,
                "deterministic_validation_error": (
                    previous_error
                ),
                "repair_instruction": (
                    "Return a complete replacement CandidateSet. "
                    "Preserve autonomous scientific freedom, but repair "
                    "the structural/schema failure. Return 3 to 6 complete "
                    "candidates. Do not emit commentary, fragments, "
                    "placeholders, duplicate candidate fields, empty "
                    "required fields, embedded candidate fields inside "
                    "hypotheses, missing evidence-ID lists, or nonpositive "
                    "estimated_model_calls. Every candidate must satisfy "
                    "the declared CandidateSet schema independently."
                ),
            }

        try:
            result = await Runner.run(
                CANDIDATE_GENERATOR,
                json.dumps(
                    attempt_payload,
                    ensure_ascii=False,
                ),
            )

            output = result.final_output

            if not isinstance(
                output,
                CandidateSet,
            ):
                raise TypeError(
                    "Candidate generation returned "
                    f"{type(output).__name__}, expected "
                    "CandidateSet"
                )

            validation = _validate_candidate_set(
                output
            )

            _write_json(
                attempts_dir
                / (
                    "candidate_generation_attempt_"
                    f"{attempt:02d}_status.json"
                ),
                {
                    "status": "passed",
                    "attempt": attempt,
                    "candidate_validation": validation,
                },
            )

            return output

        except Exception as exc:
            last_error = exc

            full_error = str(exc)

            _write_json(
                attempts_dir
                / (
                    "candidate_generation_attempt_"
                    f"{attempt:02d}_status.json"
                ),
                {
                    "status": "failed",
                    "attempt": attempt,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": full_error,
                },
            )

            # Do not feed an enormous malformed model response back into
            # the next attempt. The tail normally contains the concrete
            # Pydantic/JSON validation failures.
            previous_error = full_error[-6000:]

            if attempt >= attempts:
                break

            delay_seconds = (
                5 * (2 ** (attempt - 1))
            )

            print(
                "Candidate generation failed on "
                f"attempt {attempt}/{attempts}: "
                f"{previous_error}"
            )
            print(
                "Retrying candidate generation with "
                "deterministic repair feedback in "
                f"{delay_seconds} seconds..."
            )

            await asyncio.sleep(
                delay_seconds
            )

    raise RuntimeError(
        "Candidate generation failed after "
        f"{attempts} repair-aware attempts"
    ) from last_error


def _compact_record(
    record: LiteratureRecord,
) -> dict[str, Any]:
    abstract = record.abstract

    if abstract:
        abstract = abstract[
            :MAX_ABSTRACT_CHARACTERS
        ]

    return {
        "record_id": record.record_id,
        "title": record.title,
        "abstract": abstract,
        "publication_year": (
            record.publication_year
        ),
        "doi": record.doi,
        "source_api": record.source_api,
        "cited_by_count": (
            record.cited_by_count
        ),
        "retrieved_for_queries": (
            record.retrieved_for_queries
        ),
    }


def _supplied_synthesis_identifiers(
    records: list[dict[str, Any]],
) -> set[str]:
    """
    Return canonical bibliographic identifiers actually exposed
    to the evidence-synthesis agent.

    Only identifiers present in the compact synthesis records count
    as grounded evidence. An identifier that exists elsewhere in the
    retrieved corpus but was not supplied to synthesis must not pass.
    """
    identifiers: set[str] = set()

    for record in records:
        for field in (
            "record_id",
            "doi",
        ):
            normalised = normalise_evidence_id(
                record.get(field)
            )

            if normalised:
                identifiers.add(normalised)

    return identifiers


def _synthesis_grounding_issues(
    synthesis: EvidenceSynthesis,
    supplied_records: list[dict[str, Any]],
) -> list[str]:
    """
    Return deterministic failures for evidence identifiers that were
    not present in the records supplied to evidence synthesis.
    """
    allowed_ids = _supplied_synthesis_identifiers(
        supplied_records
    )

    issues: list[str] = []

    for section_name in (
        "established_findings",
        "unresolved_questions",
        "candidate_gaps",
    ):
        claims = getattr(
            synthesis,
            section_name,
        )

        for claim in claims:
            for evidence_id in claim.evidence_record_ids:
                canonical = normalise_evidence_id(
                    evidence_id
                )

                if (
                    canonical is None
                    or canonical not in allowed_ids
                ):
                    issues.append(
                        f"{section_name}:{claim.claim_id}: "
                        "evidence_record_id "
                        f"{evidence_id!r} was not supplied "
                        "to evidence synthesis."
                    )

    return sorted(set(issues))


def _rank_records(
    records: list[LiteratureRecord],
) -> list[LiteratureRecord]:
    """
    Apply a deterministic pre-synthesis ranking.

    Prefer records with abstracts, newer publication
    years, and larger citation counts. This is a
    retrieval-ranking rule, not a scientific conclusion.
    """
    return sorted(
        records,
        key=lambda record: (
            record.abstract is not None,
            record.publication_year or 0,
            record.cited_by_count or 0,
        ),
        reverse=True,
    )


def _validate_candidate_set(
    candidates: CandidateSet,
) -> dict[str, Any]:
    """
    Validate the full generated candidate set before
    criticism or selection.

    Candidate-level semantic validation is primarily
    enforced by the AutonomousCandidate Pydantic schema.
    This function adds set-level checks.
    """
    candidate_count = len(
        candidates.candidates
    )

    if not 3 <= candidate_count <= 6:
        raise ValueError(
            "Candidate generator must return between "
            "3 and 6 candidates; received "
            f"{candidate_count}."
        )

    candidate_ids = [
        candidate.candidate_id
        for candidate in candidates.candidates
    ]

    if len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError(
            "Candidate IDs must be unique."
        )

    for candidate in candidates.candidates:
        candidate.__class__.model_validate(
            candidate.model_dump()
        )

    return {
        "candidate_validation_status": (
            "passed"
        ),
        "candidate_count": candidate_count,
        "validated_candidate_ids": (
            candidate_ids
        ),
    }


def _remove_invalid_selection_outputs(
    *,
    selection_dir: Path,
    run_dir: Path,
) -> None:
    """
    Remove candidate-dependent artefacts while preserving
    query planning, retrieved literature, and evidence
    synthesis.
    """
    paths = [
        selection_dir / "candidates.json",
        selection_dir
        / "candidate_validation.json",
        selection_dir / "critic_reviews.json",
        selection_dir / "decision.json",
        run_dir / "state.json",
    ]

    for path in paths:
        path.unlink(
            missing_ok=True,
        )


def _validate_review_coverage(
    *,
    candidates: CandidateSet,
    reviews: ReviewSet,
) -> None:
    generated_ids = {
        candidate.candidate_id
        for candidate in candidates.candidates
    }

    reviewed_ids = {
        review.candidate_id
        for review in reviews.reviews
    }

    if reviewed_ids != generated_ids:
        missing_reviews = sorted(
            generated_ids - reviewed_ids
        )
        unknown_reviews = sorted(
            reviewed_ids - generated_ids
        )

        raise ValueError(
            "Critic reviews do not cover exactly "
            "the validated candidate set. "
            f"Missing reviews: {missing_reviews}; "
            f"unknown review IDs: {unknown_reviews}."
        )


class AutonomousDiscoveryPipeline:
    def __init__(
        self,
        *,
        model: str = "gpt-5-mini",
        per_source_per_query: int = 8,
        max_synthesis_records: int = (
            MAX_SYNTHESIS_RECORDS
        ),
    ) -> None:
        if per_source_per_query <= 0:
            raise ValueError(
                "per_source_per_query must be positive"
            )

        if max_synthesis_records <= 0:
            raise ValueError(
                "max_synthesis_records must be positive"
            )

        self.model = model
        self.per_source_per_query = (
            per_source_per_query
        )
        self.max_synthesis_records = (
            max_synthesis_records
        )

        for agent in (
            QUERY_PLANNER,
            EVIDENCE_SYNTHESISER,
            CANDIDATE_GENERATOR,
            CANDIDATE_CRITIC,
            SELECTION_JUDGE,
        ):
            agent.model = model

    async def run(
        self,
        *,
        programme: dict[str, Any],
        run_dir: Path,
    ) -> SelectionDecision:
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        literature_dir = (
            run_dir / "literature"
        )
        selection_dir = (
            run_dir / "selection"
        )

        _write_json(
            run_dir
            / "programme_snapshot.json",
            programme,
        )

        # -------------------------------------------------
        # Query planning
        # -------------------------------------------------

        query_plan_path = (
            literature_dir
            / "query_plan.json"
        )

        if query_plan_path.exists():
            print(
                "Resuming from existing "
                "query plan:",
                query_plan_path,
            )

            query_plan = (
                QueryPlan.model_validate(
                    _read_json(
                        query_plan_path
                    )
                )
            )

        else:
            query_plan = (
                await _run_agent_with_retry(
                    QUERY_PLANNER,
                    {
                        "programme": programme,
                    },
                    expected_type=QueryPlan,
                    stage_name=(
                        "Query planning"
                    ),
                )
            )

            _write_json(
                query_plan_path,
                query_plan,
            )

        # -------------------------------------------------
        # Literature retrieval
        # -------------------------------------------------

        records_path = (
            literature_dir
            / "records.json"
        )

        if records_path.exists():
            print(
                "Resuming from existing "
                "literature records:",
                records_path,
            )

            records = [
                LiteratureRecord.model_validate(
                    item
                )
                for item in _read_json(
                    records_path
                )
            ]

        else:
            records = discover_literature(
                [
                    item.query
                    for item
                    in query_plan.queries
                ],
                per_source_per_query=(
                    self
                    .per_source_per_query
                ),
            )

            _write_json(
                records_path,
                [
                    record.model_dump()
                    for record in records
                ],
            )

        # -------------------------------------------------
        # Bounded evidence-synthesis input
        # -------------------------------------------------

        ranked_records = _rank_records(
            records
        )

        synthesis_records = ranked_records[
            :self.max_synthesis_records
        ]

        compact_records = [
            _compact_record(record)
            for record
            in synthesis_records
        ]

        synthesis_input_path = (
            literature_dir
            / "synthesis_input.json"
        )

        _write_json(
            synthesis_input_path,
            {
                "total_retrieved_record_count": (
                    len(records)
                ),
                "synthesis_record_count": (
                    len(compact_records)
                ),
                "maximum_abstract_characters": (
                    MAX_ABSTRACT_CHARACTERS
                ),
                "records": compact_records,
            },
        )

        print(
            "Retrieved records:",
            len(records),
        )
        print(
            "Records supplied to synthesis:",
            len(compact_records),
        )

        # -------------------------------------------------
        # Evidence synthesis
        # -------------------------------------------------

        synthesis_path = (
            literature_dir
            / "evidence_synthesis.json"
        )

        synthesis_attempts_dir = (
            literature_dir
            / "synthesis_attempts"
        )
        synthesis_attempts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        synthesis: EvidenceSynthesis | None = None
        previous_synthesis: EvidenceSynthesis | None = None
        previous_issues: list[str] = []

        if synthesis_path.exists():
            print(
                "Resuming from existing "
                "evidence synthesis:",
                synthesis_path,
            )

            cached_synthesis = (
                EvidenceSynthesis
                .model_validate(
                    _read_json(
                        synthesis_path
                    )
                )
            )

            cached_issues = (
                _synthesis_grounding_issues(
                    cached_synthesis,
                    compact_records,
                )
            )

            if not cached_issues:
                synthesis = cached_synthesis
            else:
                print(
                    "Cached evidence synthesis failed "
                    "deterministic grounding validation."
                )
                previous_synthesis = (
                    cached_synthesis
                )
                previous_issues = (
                    cached_issues
                )

        if synthesis is None:
            for attempt in range(
                1,
                MAX_AGENT_ATTEMPTS + 1,
            ):
                payload: dict[str, Any] = {
                    "programme": programme,
                    "query_plan": (
                        query_plan.model_dump()
                    ),
                    "records": compact_records,
                }

                if previous_synthesis is not None:
                    payload[
                        "rejected_previous_synthesis"
                    ] = (
                        previous_synthesis
                        .model_dump()
                    )
                    payload[
                        "deterministic_grounding_issues"
                    ] = previous_issues
                    payload[
                        "repair_instruction"
                    ] = (
                        "Repair the evidence synthesis. "
                        "Every evidence_record_id must be "
                        "copied exactly from a record_id "
                        "or DOI present in the supplied "
                        "records. Do not introduce, infer, "
                        "reconstruct, search for, or guess "
                        "bibliographic identifiers. Remove "
                        "or rewrite claims that cannot be "
                        "supported using supplied records."
                    )

                candidate_synthesis = (
                    await _run_agent_with_retry(
                        EVIDENCE_SYNTHESISER,
                        payload,
                        expected_type=(
                            EvidenceSynthesis
                        ),
                        stage_name=(
                            "Evidence synthesis"
                        ),
                    )
                )

                attempt_path = (
                    synthesis_attempts_dir
                    / (
                        "evidence_synthesis_"
                        f"attempt_{attempt:02d}.json"
                    )
                )
                _write_json(
                    attempt_path,
                    candidate_synthesis,
                )

                grounding_issues = (
                    _synthesis_grounding_issues(
                        candidate_synthesis,
                        compact_records,
                    )
                )

                _write_json(
                    synthesis_attempts_dir
                    / (
                        "evidence_synthesis_"
                        f"attempt_{attempt:02d}_"
                        "grounding.json"
                    ),
                    {
                        "status": (
                            "passed"
                            if not grounding_issues
                            else "failed"
                        ),
                        "issue_count": len(
                            grounding_issues
                        ),
                        "issues": grounding_issues,
                    },
                )

                if not grounding_issues:
                    synthesis = (
                        candidate_synthesis
                    )
                    _write_json(
                        synthesis_path,
                        synthesis,
                    )
                    break

                print(
                    "Evidence synthesis grounding "
                    f"failed on attempt {attempt}/"
                    f"{MAX_AGENT_ATTEMPTS}: "
                    f"{len(grounding_issues)} "
                    "issue(s)."
                )

                previous_synthesis = (
                    candidate_synthesis
                )
                previous_issues = (
                    grounding_issues
                )

            if synthesis is None:
                _write_json(
                    literature_dir
                    / "synthesis_grounding.json",
                    {
                        "status": "failed",
                        "issue_count": len(
                            previous_issues
                        ),
                        "issues": (
                            previous_issues
                        ),
                    },
                )

                raise RuntimeError(
                    "Evidence synthesis could not "
                    "be grounded exclusively in "
                    "supplied literature records "
                    "after bounded autonomous repair."
                )

        _write_json(
            literature_dir
            / "synthesis_grounding.json",
            {
                "status": "passed",
                "issue_count": 0,
                "issues": [],
            },
        )

        # -------------------------------------------------
        # Candidate generation and validation
        # -------------------------------------------------

        candidates_path = (
            selection_dir
            / "candidates.json"
        )

        candidate_validation: dict[
            str,
            Any,
        ]

        if candidates_path.exists():
            print(
                "Resuming from existing "
                "candidates:",
                candidates_path,
            )

            try:
                candidates = (
                    CandidateSet
                    .model_validate(
                        _read_json(
                            candidates_path
                        )
                    )
                )

                candidate_validation = (
                    _validate_candidate_set(
                        candidates
                    )
                )

            except Exception as exc:
                print(
                    "Cached candidates failed "
                    "validation:",
                    exc,
                )
                print(
                    "Deleting invalid candidate "
                    "and downstream selection "
                    "artefacts."
                )

                _remove_invalid_selection_outputs(
                    selection_dir=selection_dir,
                    run_dir=run_dir,
                )

                candidates = (
                    await _generate_candidate_set_with_repair(
                        payload={
                            "programme": (
                                programme
                            ),
                            "evidence_synthesis": (
                                synthesis
                                .model_dump()
                            ),
                        },
                        attempts_dir=(
                            selection_dir
                            / "candidate_generation_attempts"
                        ),
                    )
                )

                candidate_validation = (
                    _validate_candidate_set(
                        candidates
                    )
                )

                _write_json(
                    candidates_path,
                    candidates,
                )

        else:
            candidates = (
                await _generate_candidate_set_with_repair(
                    payload={
                        "programme": programme,
                        "evidence_synthesis": (
                            synthesis
                            .model_dump()
                        ),
                    },
                    attempts_dir=(
                        selection_dir
                        / "candidate_generation_attempts"
                    ),
                )
            )

            candidate_validation = (
                _validate_candidate_set(
                    candidates
                )
            )

            _write_json(
                candidates_path,
                candidates,
            )

        candidate_validation_path = (
            selection_dir
            / "candidate_validation.json"
        )

        _write_json(
            candidate_validation_path,
            candidate_validation,
        )

        print(
            "Candidate validation passed:",
            candidate_validation[
                "candidate_count"
            ],
            "candidates",
        )

        # -------------------------------------------------
        # Candidate criticism
        # -------------------------------------------------

        reviews_path = (
            selection_dir
            / "critic_reviews.json"
        )

        if reviews_path.exists():
            print(
                "Resuming from existing "
                "critic reviews:",
                reviews_path,
            )

            try:
                reviews = (
                    ReviewSet.model_validate(
                        _read_json(
                            reviews_path
                        )
                    )
                )

                _validate_review_coverage(
                    candidates=candidates,
                    reviews=reviews,
                )

            except Exception as exc:
                print(
                    "Cached critic reviews "
                    "failed validation:",
                    exc,
                )
                print(
                    "Regenerating critic "
                    "reviews and selection."
                )

                reviews_path.unlink(
                    missing_ok=True,
                )
                (
                    selection_dir
                    / "decision.json"
                ).unlink(
                    missing_ok=True,
                )
                (
                    run_dir
                    / "state.json"
                ).unlink(
                    missing_ok=True,
                )

                reviews = (
                    await _run_agent_with_retry(
                        CANDIDATE_CRITIC,
                        {
                            "programme": (
                                programme
                            ),
                            "evidence_synthesis": (
                                synthesis
                                .model_dump()
                            ),
                            "candidates": (
                                candidates
                                .model_dump()
                            ),
                        },
                        expected_type=(
                            ReviewSet
                        ),
                        stage_name=(
                            "Candidate criticism"
                        ),
                    )
                )

                _validate_review_coverage(
                    candidates=candidates,
                    reviews=reviews,
                )

                _write_json(
                    reviews_path,
                    reviews,
                )

        else:
            reviews = (
                await _run_agent_with_retry(
                    CANDIDATE_CRITIC,
                    {
                        "programme": programme,
                        "evidence_synthesis": (
                            synthesis
                            .model_dump()
                        ),
                        "candidates": (
                            candidates
                            .model_dump()
                        ),
                    },
                    expected_type=ReviewSet,
                    stage_name=(
                        "Candidate criticism"
                    ),
                )
            )

            _validate_review_coverage(
                candidates=candidates,
                reviews=reviews,
            )

            _write_json(
                reviews_path,
                reviews,
            )

        # -------------------------------------------------
        # Evidence-grounded selection
        # -------------------------------------------------

        decision_path = (
            selection_dir
            / "decision.json"
        )

        if decision_path.exists():
            print(
                "Resuming from existing "
                "selection decision:",
                decision_path,
            )

            try:
                decision = (
                    SelectionDecision
                    .model_validate(
                        _read_json(
                            decision_path
                        )
                    )
                )

            except Exception as exc:
                print(
                    "Cached selection decision "
                    "failed validation:",
                    exc,
                )
                print(
                    "Regenerating selection "
                    "decision."
                )

                decision_path.unlink(
                    missing_ok=True,
                )

                decision = (
                    await _run_agent_with_retry(
                        SELECTION_JUDGE,
                        {
                            "programme": (
                                programme
                            ),
                            "evidence_synthesis": (
                                synthesis
                                .model_dump()
                            ),
                            "candidates": (
                                candidates
                                .model_dump()
                            ),
                            "critic_reviews": (
                                reviews
                                .model_dump()
                            ),
                        },
                        expected_type=(
                            SelectionDecision
                        ),
                        stage_name=(
                            "Candidate selection"
                        ),
                    )
                )

                _write_json(
                    decision_path,
                    decision,
                )

        else:
            decision = (
                await _run_agent_with_retry(
                    SELECTION_JUDGE,
                    {
                        "programme": programme,
                        "evidence_synthesis": (
                            synthesis
                            .model_dump()
                        ),
                        "candidates": (
                            candidates
                            .model_dump()
                        ),
                        "critic_reviews": (
                            reviews
                            .model_dump()
                        ),
                    },
                    expected_type=(
                        SelectionDecision
                    ),
                    stage_name=(
                        "Candidate selection"
                    ),
                )
            )

            _write_json(
                decision_path,
                decision,
            )

        # -------------------------------------------------
        # Final integrity gates
        # -------------------------------------------------

        candidate_ids = {
            candidate.candidate_id
            for candidate
            in candidates.candidates
        }

        if (
            decision.selected_candidate_id
            not in candidate_ids
        ):
            raise ValueError(
                "Selection judge chose an "
                "unknown candidate ID."
            )

        if (
            decision.runner_up_candidate_id
            is not None
            and decision.runner_up_candidate_id
            not in candidate_ids
        ):
            raise ValueError(
                "Selection judge named an "
                "unknown runner-up candidate ID."
            )

        if (
            decision.runner_up_candidate_id
            == decision.selected_candidate_id
        ):
            raise ValueError(
                "Selected candidate and runner-up "
                "must be different."
            )

        validation_status = (
            candidate_validation.get(
                "candidate_validation_status"
            )
        )

        if validation_status != "passed":
            raise ValueError(
                "Cannot enter "
                "AUTONOMOUS_DESIGN_SELECTED "
                "without successful candidate "
                "validation."
            )

        _validate_review_coverage(
            candidates=candidates,
            reviews=reviews,
        )

        # -------------------------------------------------
        # Final state
        # -------------------------------------------------

        _write_json(
            run_dir / "state.json",
            {
                "state": (
                    "AUTONOMOUS_DESIGN_SELECTED"
                ),
                "selected_candidate_id": (
                    decision
                    .selected_candidate_id
                ),
                "runner_up_candidate_id": (
                    decision
                    .runner_up_candidate_id
                ),
                "candidate_validation_status": (
                    validation_status
                ),
                "validated_candidate_count": (
                    candidate_validation[
                        "candidate_count"
                    ]
                ),
                "required_repairs_before_preregistration": (
                    decision
                    .required_repairs_before_preregistration
                ),
            },
        )

        return decision