from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cnsm_agentic.autonomous_research.final_agents import (
    MANUSCRIPT_REVISER,
)
from cnsm_agentic.autonomous_research.final_pipeline import (
    _manuscript_text,
    audit_manuscript_artifact_references,
    audit_manuscript_publication_sanity,
    run_agent,
)
from cnsm_agentic.autonomous_research.final_schemas import (
    ManuscriptPackage,
    PreregistrationDocument,
)
from cnsm_agentic.autonomous_research.final_guardrails import (
    sha256_file,
)
from cnsm_agentic.autonomous_research.publication_renderer import (
    build_publication_artifacts,
)


SOURCE_SCIENTIFIC_FILES = (
    "literature/records.json",
    "literature/evidence_synthesis.json",
    "preregistration/preregistration.json",
    "execution/execution_manifest.json",
    "analysis/results.json",
    "analysis/deterministic_reconciliation.json",
    "manuscript/manuscript_evidence_bundle.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def extract_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        for key in (
            "records",
            "verified_records",
            "items",
        ):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate

    raise ValueError(
        "Could not extract verified literature records "
        "from literature/records.json."
    )


def scientific_hashes(
    source_run: Path,
) -> dict[str, str]:
    hashes: dict[str, str] = {}

    for relative in SOURCE_SCIENTIFIC_FILES:
        path = source_run / relative

        if not path.exists():
            raise FileNotFoundError(
                f"Required archived scientific artifact missing: {path}"
            )

        hashes[relative] = sha256_file(path)

    return hashes


def citation_ids(
    manuscript: ManuscriptPackage,
) -> list[str]:
    value = manuscript.model_dump().get(
        "cited_record_ids",
        [],
    )

    if not isinstance(value, list):
        raise ValueError(
            "Manuscript cited_record_ids is not a list."
        )

    return [str(item) for item in value]


def recovery_instruction(
    *,
    attempt: int,
    attempts: int,
    page_count: int,
    maximum_pages: int,
) -> str:
    return (
        "POST-LOCK PUBLICATION-ONLY RECOVERY. "
        "The autonomous scientific execution is complete and immutable. "
        "This operation must not perform or imply any new literature search, "
        "research-question selection, design change, preregistration change, "
        "experiment, model evaluation, analysis, result generation, or "
        "scientific outcome selection. "
        "\n\n"
        f"This is bounded recovery attempt {attempt} of {attempts}. "
        f"The current clean manuscript occupies {page_count} compiled "
        f"IEEE page(s) of the required {maximum_pages}. "
        "\n\n"
        "SCIENTIFIC CONTENT IS FROZEN. Build cumulatively on the supplied "
        "manuscript using ONLY facts already supported by the supplied "
        "verified literature, frozen preregistration, completed execution "
        "manifest, archived analysis results, deterministic reconciliation, "
        "and manuscript evidence bundle. "
        "\n\n"
        "Preserve all existing supported scientific substance. Do not "
        "shorten or replace supported passages merely for stylistic variety. "
        "Add multiple evidence-grounded substantive explanations where "
        "useful so that the manuscript reaches exactly the required five "
        "compiled IEEE pages. "
        "\n\n"
        "Appropriate additions, only when directly supported by the frozen "
        "evidence, include deeper related-work synthesis; experimental-design "
        "rationale; explanation of the paired estimand; interpretation of "
        "already observed effect sizes and discordant pairs; failure-mode "
        "discussion based on archived outcomes; uncertainty and power "
        "interpretation; threats to validity; limitations; reproducibility "
        "explanation; and qualified operational implications. "
        "\n\n"
        "Do not introduce or change the research question, hypotheses, "
        "preregistration, study design, execution, sample sizes, numerical "
        "results, statistical tests, effect sizes, confidence intervals, "
        "p-values, interpretation direction, scientific conclusions, "
        "citations, citation IDs, evidence records, experiments, analyses, "
        "model calls, data, or results. "
        "\n\n"
        "Do not add generic filler, repeated provenance, artifact inventories, "
        "filesystem-path lists, additional hashes, DOI labels, raw commands, "
        "reviewer-response language, formatting tricks, artificial spacing, "
        "or template manipulation. "
        "\n\n"
        "Perform a final content-preserving proofreading pass for obvious "
        "copy defects such as accidentally concatenated ordinary words, "
        "missing spaces after punctuation, duplicated adjacent words, and "
        "malformed encoded text. "
        "\n\n"
        "The sole publication objective is a scientifically unchanged, "
        "evidence-grounded manuscript that passes all deterministic audits "
        "and occupies exactly five IEEE pages."
    )


async def recover(
    *,
    source_run: Path,
    output_dir: Path,
    model: str,
    maximum_attempts: int,
    dry_run: bool,
) -> int:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()

    freeze_manifest_path = (
        source_run
        / "provenance"
        / "freeze_manifest.json"
    )

    if not freeze_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing real-run freeze manifest: {freeze_manifest_path}"
        )

    source_freeze = read_json(
        freeze_manifest_path
    )

    if source_freeze.get("development_rehearsal") is not False:
        raise RuntimeError(
            "Publication recovery requires a real non-rehearsal source run."
        )

    source_hashes_before = scientific_hashes(
        source_run
    )

    current_manuscript = ManuscriptPackage.model_validate(
        read_json(
            source_run
            / "manuscript"
            / "revised_package.json"
        )
    )

    original_citation_ids = citation_ids(
        current_manuscript
    )

    records = extract_records(
        read_json(
            source_run
            / "literature"
            / "records.json"
        )
    )

    evidence_synthesis = read_json(
        source_run
        / "literature"
        / "evidence_synthesis.json"
    )

    execution_manifest = read_json(
        source_run
        / "execution"
        / "execution_manifest.json"
    )

    manuscript_evidence_bundle = read_json(
        source_run
        / "manuscript"
        / "manuscript_evidence_bundle.json"
    )

    preregistration = PreregistrationDocument.model_validate(
        read_json(
            source_run
            / "preregistration"
            / "preregistration.json"
        )
    )

    analysis_results = read_json(
        source_run
        / "analysis"
        / "results.json"
    )

    deterministic_reconciliation = read_json(
        source_run
        / "analysis"
        / "deterministic_reconciliation.json"
    )

    paper_run_constraints = read_json(
        source_run
        / "provenance"
        / "paper_run_constraints.json"
    )

    publication_dir = (
        output_dir
        / "manuscript"
        / "final"
    )

    revision_dir = (
        output_dir
        / "manuscript"
        / "revision_rounds"
    )

    publication_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    revision_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recovery_commit = git(
        "rev-parse",
        "HEAD",
    )

    recovery_manifest = {
        "schema_version": "1.0",
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "recovery_scope": (
            "manuscript_publication_only"
        ),
        "source_scientific_run": str(
            source_run
        ),
        "source_framework_commit": (
            source_freeze.get(
                "framework_commit"
            )
        ),
        "source_framework_tag": (
            source_freeze.get(
                "framework_tag"
            )
        ),
        "recovery_framework_commit": (
            recovery_commit
        ),
        "model": model,
        "scientific_execution_reused": True,
        "scientific_execution_rerun": False,
        "literature_retrieval_rerun": False,
        "design_rerun": False,
        "preregistration_rerun": False,
        "experiment_rerun": False,
        "analysis_rerun": False,
        "source_scientific_artifact_sha256": (
            source_hashes_before
        ),
        "maximum_recovery_attempts": (
            maximum_attempts
        ),
        "dry_run": dry_run,
    }

    write_json(
        output_dir
        / "recovery_manifest.json",
        recovery_manifest,
    )

    # Establish the archived cleaned manuscript as the recovery seed.
    best_manuscript = current_manuscript

    validation = build_publication_artifacts(
        manuscript=best_manuscript.model_dump(),
        verified_records=records,
        output_dir=publication_dir,
        paper_run_constraints=paper_run_constraints,
    )

    sanity = audit_manuscript_publication_sanity(
        run_dir=output_dir,
    )

    # Artifact claims refer to the original scientific run, not
    # the publication-recovery workspace.
    artifact_audit = (
        audit_manuscript_artifact_references(
            manuscript=best_manuscript,
            run_dir=source_run,
        )
    )

    write_json(
        publication_dir
        / "publication_validation_seed.json",
        validation,
    )
    write_json(
        publication_dir
        / "publication_sanity_audit_seed.json",
        sanity,
    )
    write_json(
        publication_dir
        / "artifact_reference_audit_seed.json",
        artifact_audit,
    )

    seed_page_count = validation.get(
        "page_count"
    )
    maximum_pages = validation.get(
        "maximum_pages"
    )

    if validation.get("compile_status") != "passed":
        raise RuntimeError(
            "Archived cleaned manuscript does not compile "
            "under the recovery framework."
        )

    if sanity.get("passed") is not True:
        raise RuntimeError(
            "Archived cleaned manuscript still fails "
            f"publication sanity: {sanity.get('issues')}"
        )

    if artifact_audit.get("passed") is not True:
        raise RuntimeError(
            "Archived cleaned manuscript fails artifact-reference "
            f"audit: {artifact_audit.get('issues')}"
        )

    if not isinstance(seed_page_count, int):
        raise RuntimeError(
            "Seed manuscript page count is unavailable."
        )

    if not isinstance(maximum_pages, int):
        raise RuntimeError(
            "Maximum page count is unavailable."
        )

    best_page_count = seed_page_count
    best_validation = validation

    write_json(
        output_dir
        / "manuscript"
        / "revised_package.json",
        best_manuscript,
    )

    if (
        best_page_count == maximum_pages
        or dry_run
    ):
        source_hashes_after = scientific_hashes(
            source_run
        )

        if source_hashes_after != source_hashes_before:
            raise RuntimeError(
                "Source scientific artifacts changed during recovery."
            )

        print(
            "RECOVERY DRY RUN PASS"
            if dry_run
            else "RECOVERY ALREADY EXACT-PAGE"
        )
        print(
            f"Seed pages: {best_page_count}/{maximum_pages}"
        )
        return 0

    MANUSCRIPT_REVISER.model = model

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        instruction = recovery_instruction(
            attempt=attempt,
            attempts=maximum_attempts,
            page_count=best_page_count,
            maximum_pages=maximum_pages,
        )

        candidate = await run_agent(
            MANUSCRIPT_REVISER,
            {
                "current_manuscript": (
                    best_manuscript.model_dump()
                ),
                "verified_records": records,
                "execution_manifest": (
                    execution_manifest
                ),
                "manuscript_evidence_bundle": (
                    manuscript_evidence_bundle
                ),
                "evidence_synthesis": (
                    evidence_synthesis
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
                "publication_validation": (
                    best_validation
                ),
                "paper_run_constraints": (
                    paper_run_constraints
                ),
                "revision_mode": (
                    "post_lock_publication_only_"
                    "underfill_recovery"
                ),
                "revision_instruction": (
                    instruction
                ),
            },
            expected_type=ManuscriptPackage,
            stage_name=(
                "Post-lock publication-only recovery "
                f"attempt {attempt}"
            ),
        )

        write_json(
            revision_dir
            / f"candidate_{attempt:02d}.json",
            candidate,
        )

        # Citation identity is a hard post-lock invariant.
        if citation_ids(candidate) != original_citation_ids:
            write_json(
                revision_dir
                / f"candidate_{attempt:02d}_rejected.json",
                {
                    "reason": (
                        "cited_record_ids changed"
                    )
                },
            )
            continue

        candidate_validation = (
            build_publication_artifacts(
                manuscript=candidate.model_dump(),
                verified_records=records,
                output_dir=publication_dir,
                paper_run_constraints=(
                    paper_run_constraints
                ),
            )
        )

        candidate_sanity = (
            audit_manuscript_publication_sanity(
                run_dir=output_dir,
            )
        )

        candidate_artifact_audit = (
            audit_manuscript_artifact_references(
                manuscript=candidate,
                run_dir=source_run,
            )
        )

        write_json(
            publication_dir
            / (
                "publication_validation_"
                f"recovery_{attempt:02d}.json"
            ),
            candidate_validation,
        )
        write_json(
            publication_dir
            / (
                "publication_sanity_audit_"
                f"recovery_{attempt:02d}.json"
            ),
            candidate_sanity,
        )
        write_json(
            publication_dir
            / (
                "artifact_reference_audit_"
                f"recovery_{attempt:02d}.json"
            ),
            candidate_artifact_audit,
        )

        candidate_page_count = (
            candidate_validation.get(
                "page_count"
            )
        )

        current_text_length = len(
            _manuscript_text(
                best_manuscript
            )
        )

        candidate_text_length = len(
            _manuscript_text(
                candidate
            )
        )

        candidate_is_clean_non_regressing = (
            candidate_validation.get(
                "compile_status"
            )
            == "passed"
            and candidate_sanity.get(
                "passed"
            )
            is True
            and candidate_artifact_audit.get(
                "passed"
            )
            is True
            and isinstance(
                candidate_page_count,
                int,
            )
            and candidate_page_count
            <= maximum_pages
            and candidate_page_count
            >= best_page_count
            and candidate_text_length
            > current_text_length
        )

        if not candidate_is_clean_non_regressing:
            write_json(
                revision_dir
                / f"candidate_{attempt:02d}_rejected.json",
                {
                    "page_count": (
                        candidate_page_count
                    ),
                    "compile_status": (
                        candidate_validation.get(
                            "compile_status"
                        )
                    ),
                    "sanity_passed": (
                        candidate_sanity.get(
                            "passed"
                        )
                    ),
                    "artifact_audit_passed": (
                        candidate_artifact_audit.get(
                            "passed"
                        )
                    ),
                    "candidate_text_length": (
                        candidate_text_length
                    ),
                    "current_text_length": (
                        current_text_length
                    ),
                },
            )
            continue

        best_manuscript = candidate
        best_validation = (
            candidate_validation
        )
        best_page_count = (
            candidate_page_count
        )

        write_json(
            output_dir
            / "manuscript"
            / "revised_package.json",
            best_manuscript,
        )

        write_json(
            revision_dir
            / "best_recovered_package.json",
            best_manuscript,
        )

        if best_page_count == maximum_pages:
            break

    # Always re-render the exact selected candidate.
    final_validation = build_publication_artifacts(
        manuscript=best_manuscript.model_dump(),
        verified_records=records,
        output_dir=publication_dir,
        paper_run_constraints=paper_run_constraints,
    )

    final_sanity = audit_manuscript_publication_sanity(
        run_dir=output_dir,
    )

    final_artifact_audit = (
        audit_manuscript_artifact_references(
            manuscript=best_manuscript,
            run_dir=source_run,
        )
    )

    write_json(
        publication_dir
        / "publication_validation_final_authoritative.json",
        final_validation,
    )
    write_json(
        publication_dir
        / "publication_sanity_audit_final_authoritative.json",
        final_sanity,
    )
    write_json(
        publication_dir
        / "artifact_reference_audit_final_authoritative.json",
        final_artifact_audit,
    )

    source_hashes_after = scientific_hashes(
        source_run
    )

    if source_hashes_after != source_hashes_before:
        raise RuntimeError(
            "Source scientific artifacts changed during recovery."
        )

    exact_page_success = (
        final_validation.get(
            "compile_status"
        )
        == "passed"
        and final_validation.get(
            "page_count"
        )
        == final_validation.get(
            "maximum_pages"
        )
        and final_sanity.get(
            "passed"
        )
        is True
        and final_artifact_audit.get(
            "passed"
        )
        is True
    )

    recovery_manifest[
        "completed_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    recovery_manifest[
        "final_page_count"
    ] = final_validation.get(
        "page_count"
    )

    recovery_manifest[
        "publication_recovery_passed"
    ] = exact_page_success

    recovery_manifest[
        "source_scientific_artifacts_unchanged"
    ] = (
        source_hashes_after
        == source_hashes_before
    )

    write_json(
        output_dir
        / "recovery_manifest.json",
        recovery_manifest,
    )

    print(
        "Publication recovery:",
        "PASS"
        if exact_page_success
        else "FAILED",
    )
    print(
        "Final pages:",
        final_validation.get(
            "page_count"
        ),
        "/",
        final_validation.get(
            "maximum_pages"
        ),
    )

    return (
        0
        if exact_page_success
        else 2
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover only the final manuscript/publication "
            "stage of an already completed autonomous run."
        )
    )

    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
    )

    parser.add_argument(
        "--maximum-attempts",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise SystemExit(
        asyncio.run(
            recover(
                source_run=args.source_run,
                output_dir=args.output_dir,
                model=args.model,
                maximum_attempts=(
                    args.maximum_attempts
                ),
                dry_run=args.dry_run,
            )
        )
    )


if __name__ == "__main__":
    main()
