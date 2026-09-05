from __future__ import annotations

import asyncio
import json
import re
from itertools import product
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
    build_evidence_alias_index,
    normalise_evidence_id,
    verify_evidence,
)
from .execution_adapters import (
    adapter_compatibility_issues,
    register_builtin_execution_adapters,
    registered_adapter_families,
    registered_adapter_planning_contracts,
    resolve_adapter,
    validate_execution_manifest,
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
from .analysis_executors import (
    register_builtin_analysis_executors,
    analysis_compatibility_issues,
    registered_analysis_families,
    registered_analysis_planning_contracts,
    resolve_analysis_executor,
    validate_analysis_results,
)
from .repair_schemas import (
    RepairedStudyDesign,
    RepairReadinessReport,
)

from .publication_renderer import (
    build_publication_artifacts,
)


T = TypeVar("T")


def _positive_capability_claim(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    """
    Detect a positive experimental capability requirement while avoiding
    obvious explicit negations such as 'no RAG', 'RAG is disabled', or
    'without independent generation'.

    This is deliberately applied only to execution-defining scientific
    fields, never to literature/background text.
    """
    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            left = text[
                max(0, match.start() - 100):
                match.start()
            ]
            right = text[
                match.end():
                min(len(text), match.end() + 100)
            ]

            negated_before = re.search(
                r"\b(?:no|without|never|not|disable(?:d)?|"
                r"exclude(?:d)?|omit(?:ted)?)\b"
                r".{0,60}$",
                left,
                flags=re.IGNORECASE,
            )

            negated_after = re.search(
                r"^\s*.{0,60}\b(?:is|are|was|were|be)?\s*"
                r"(?:not used|not enabled|disabled|excluded|omitted|"
                r"unavailable)\b",
                right,
                flags=re.IGNORECASE,
            )

            if not negated_before and not negated_after:
                return True

    return False


def repaired_design_adapter_capability_issues(
    design: dict[str, Any],
    *,
    available_adapter_contracts: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Reject a scientific design before preregistration when it positively
    requires an execution capability that no registered adapter provides.

    This gate protects scientific/execution consistency without selecting
    a preferred hypothesis or reacting to observed outcomes.
    """

    # Only fields that define the experiment itself are scanned. In
    # particular, literature/evidence/background material is not included.
    scientific_fields = (
        "research_question",
        "confirmatory_hypotheses",
        "primary_estimand",
        "secondary_estimands",
        "sampling_plan",
        "analysis_plan",
        "transformation_scope",
    )

    parts: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for item in value.values():
                add(item)

    for field in scientific_fields:
        add(design.get(field))

    science_text = "\n".join(parts)

    contracts = list(
        available_adapter_contracts.values()
    )

    def any_adapter_supports(
        field: str,
        *,
        default: bool = False,
    ) -> bool:
        return any(
            bool(contract.get(field, default))
            for contract in contracts
        )

    issues: list[str] = []

    if (
        _positive_capability_claim(
            science_text,
            (
                r"\bretrieval[- ]augmented generation\b",
                r"\bRAG\b",
                r"\bretrieval poisoning\b",
                r"\bpoison(?:ed|ing)? retrieval\b",
            ),
        )
        and not any_adapter_supports(
            "retrieval_augmented_generation"
        )
    ):
        issues.append(
            "Scientific design requires retrieval-augmented generation "
            "or retrieval poisoning, but no registered execution adapter "
            "enables retrieval_augmented_generation."
        )

    if (
        _positive_capability_claim(
            science_text,
            (
                r"\bindependent(?:ly)? generated\b",
                r"\bindependent (?:condition|arm) generation\b",
                r"\bseparate(?:ly)? generated\b",
                r"\bper-condition independent sampling\b",
                r"\bindependent per-condition sampling\b",
            ),
        )
        and not any_adapter_supports(
            "independent_condition_generation"
        )
    ):
        issues.append(
            "Scientific design requires independent per-condition "
            "generation, but no registered execution adapter enables "
            "independent_condition_generation."
        )

    if (
        _positive_capability_claim(
            science_text,
            (
                r"\bmulti[- ]model consensus\b",
                r"\bmodel ensemble\b",
                r"\bmulti[- ]model ensemble\b",
                r"\bmajority vot(?:e|ing) across models\b",
                r"\bthree[- ]model consensus\b",
                r"\b3[- ]model consensus\b",
            ),
        )
        and not any_adapter_supports(
            "supports_multi_model_consensus"
        )
    ):
        issues.append(
            "Scientific design requires multi-model consensus or an "
            "ensemble, but no registered execution adapter supports it."
        )

    if (
        _positive_capability_claim(
            science_text,
            (
                r"\bsimulated human gate\b",
                r"\bsimulated human review\b",
                r"\bhuman-in-the-loop gate\b",
                r"\bhuman gate\b",
            ),
        )
        and not any_adapter_supports(
            "supports_simulated_human_gate"
        )
    ):
        issues.append(
            "Scientific design requires a simulated-human/human gate, "
            "but no registered execution adapter supports it."
        )

    if (
        all(
            term in science_text.lower()
            for term in (
                "benign",
                "ambiguous",
                "adversarial",
            )
        )
        and not any_adapter_supports(
            "supports_prompt_family_stratification"
        )
    ):
        issues.append(
            "Scientific design requires benign/ambiguous/adversarial "
            "prompt-family stratification, but no registered execution "
            "adapter supports it."
        )

    return sorted(set(issues))



def _manuscript_text(
    manuscript: ManuscriptPackage | dict[str, Any],
) -> str:
    if hasattr(manuscript, "model_dump"):
        payload = manuscript.model_dump()
    else:
        payload = manuscript

    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    return "\n".join(parts)


def _normalise_claimed_artifact_path(
    value: str,
) -> str:
    value = value.strip().strip(
        "\"'`.,;:()[]{}"
    )

    while value.endswith(
        (".", ",", ";", ":", ")")
    ):
        value = value[:-1]

    return value



def sanitize_structured_manuscript_publication_metadata(
    manuscript: Any,
    *,
    run_dir: Path | None = None,
) -> Any:
    """Deterministically remove machine-oriented publication metadata.

    This is a non-scientific publication-hygiene transformation. It preserves
    manuscript claims and prose while replacing run-relative artifact locators
    with concise human-readable descriptions and removing inline DOI metadata.

    The mandatory Disclosure Statement and bibliography/citation identifiers
    are intentionally left unchanged.
    """
    import re

    authoritative_master_prompt_sha: str | None = None
    if run_dir is not None:
        master_prompt_sha_path = Path(run_dir) / "provenance" / "master_prompt.sha256"
        if master_prompt_sha_path.exists():
            candidate_sha = master_prompt_sha_path.read_text(encoding="utf-8").strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", candidate_sha):
                authoritative_master_prompt_sha = candidate_sha

    path_replacements = {
        "analysis/condition_summary.csv":
            "the archived condition summary",
        "analysis/paired_contingency_table.csv":
            "the archived paired contingency table",
        "analysis/results.json":
            "the archived analysis results",
        "execution/model_configuration.json":
            "the archived model configuration",
        "execution/task_manifest.jsonl":
            "the archived task manifest",
        "execution/scoring/task-00000X-*.json":
            "the archived per-episode scoring records",
        "execution/*":
            "the archived execution artifacts",
        "analysis/*":
            "the archived analysis artifacts",
    }

    artifact_path_pattern = re.compile(
        r"(?i)\b(?:execution|analysis|provenance|manuscript)"
        r"/[A-Za-z0-9_.*/-]+"
    )

    bracketed_doi_pattern = re.compile(
        r"\s*\[\s*doi\s*:\s*[^\]]+\]",
        flags=re.IGNORECASE,
    )

    inline_doi_label_pattern = re.compile(
        r"(?i)\bDOI\s*:\s*"
    )

    def sanitize_text(value: str) -> str:
        cleaned = bracketed_doi_pattern.sub("", value)

        for raw_path, replacement in path_replacements.items():
            cleaned = cleaned.replace(raw_path, replacement)

        # Any unexpected run-relative locator is converted only to a generic
        # archived-artifact description rather than deleted.
        def generic_artifact_replacement(
            match: re.Match[str],
        ) -> str:
            raw = match.group(0)
            prefix = raw.split("/", 1)[0].lower()

            labels = {
                "analysis": "the archived analysis artifact",
                "execution": "the archived execution artifact",
                "provenance": "the archived provenance artifact",
                "manuscript": "the archived manuscript artifact",
            }
            return labels[prefix]

        cleaned = artifact_path_pattern.sub(
            generic_artifact_replacement,
            cleaned,
        )

        # Defensive fallback for a DOI label that was not bracketed.
        cleaned = inline_doi_label_pattern.sub("", cleaned)

        # The renderer creates the canonical bibliography itself. Remove any
        # standalone body heading named "References", including when that
        # heading is one line inside a larger multi-line manuscript field.
        # Ordinary prose containing the word "references" is preserved.
        #
        # Supported heading forms include:
        #   References
        #   ## References
        #   REFERENCES:
        #   \section{References}
        #   \section*{References}
        references_heading_line_pattern = re.compile(
            r"(?im)"
            r"^[ \t]*"
            r"(?:"
            r"#{1,6}[ \t]*references[ \t]*:?"
            r"|references[ \t]*:?"
            r"|\\section\*?\{[ \t]*references[ \t]*\}"
            r"|\\subsection\*?\{[ \t]*references[ \t]*\}"
            r")"
            r"[ \t]*$"
            r"(?:\r?\n)?"
        )
        cleaned = references_heading_line_pattern.sub("", cleaned)

        # Peer-review workflow language must not leak into the submitted paper.
        # Remove only complete sentences explicitly framed as reviewer process;
        # scientific sentences before and after them are left untouched.
        reviewer_process_sentence_pattern = re.compile(
            r"(?i)"
            r"(?<!\w)"
            r"(?:reviewers?|the reviewers?)\s+"
            r"(?:requested|asked|recommended|suggested|required)"
            r"[^.!?]*(?:[.!?]+|$)"
            r"\s*"
        )
        cleaned = reviewer_process_sentence_pattern.sub("", cleaned)

        # Full cryptographic digests are machine provenance, not ordinary
        # scientific prose. Remove them deterministically together with an
        # immediately associated sha256/SHA-256 label. The mandatory
        # master-prompt digest in Disclosure is protected by a temporary
        # non-hex token before this function is called.
        cleaned = re.sub(
            r"(?i)"
            r"(?:\\?b?sha(?:[_\\ -]?256)?\\?s*[:=]\\?s*)?"
            r"(?<![0-9A-Fa-f])"
            r"[0-9A-Fa-f]{64}"
            r"(?![0-9A-Fa-f])",
            "",
            cleaned,
        )

        # pdfLaTeX cannot render arbitrary astral-plane Unicode such as
        # emoji embedded in titles or bibliographic metadata. Remove these
        # presentation symbols deterministically while preserving ordinary
        # Unicode text such as accented names and scientific punctuation.
        cleaned = re.sub(
            r"[\U00010000-\U0010FFFF]",
            "",
            cleaned,
        )

        # Repair only whitespace introduced by local metadata removal.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)

        return cleaned

    def sanitize_disclosure_text(value: str) -> str:
        """Sanitize Disclosure while preserving its mandatory prompt proof.

        The Disclosure Statement must retain the immutable
        provenance/master_prompt.txt locator and its associated SHA-256.
        Other machine-oriented artifact paths and full digests are
        publication metadata and are removed deterministically.
        """
        master_prompt_path = "provenance/master_prompt.txt"

        # Locate the mandatory master-prompt digest before replacing either
        # component with temporary tokens. Structured manuscript text has not
        # yet received renderer-level \\allowbreak{} insertion here.
        master_prompt_sha: str | None = None

        master_prompt_match = re.search(
            r"provenance/master_prompt\.txt"
            r".{0,300}?"
            r"(?<![0-9A-Fa-f])"
            r"([0-9A-Fa-f]{64})"
            r"(?![0-9A-Fa-f])",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if authoritative_master_prompt_sha is not None:
            master_prompt_sha = authoritative_master_prompt_sha
        elif master_prompt_match is not None:
            master_prompt_sha = master_prompt_match.group(1)

        path_token = "MASTERPROMPTPATHTOKEN"
        sha_token = "MASTERPROMPTSHA256TOKEN"

        protected = value.replace(
            master_prompt_path,
            path_token,
        )

        if master_prompt_sha is not None and master_prompt_sha in protected:
            protected = protected.replace(
                master_prompt_sha,
                sha_token,
                1,
            )

        # Apply the ordinary publication-metadata sanitizer while the
        # mandatory master-prompt provenance is protected.
        cleaned = sanitize_text(protected)

        # Remove every remaining full SHA-256 plus an immediately associated
        # textual sha256/SHA-256 label. These are secondary machine-provenance
        # digests and belong in the archived run rather than the paper.
        cleaned = re.sub(
            r"(?i)"
            r"(?:\bsha\s*-?\s*256\s*[:=]\s*)?"
            r"(?<![0-9A-Fa-f])"
            r"[0-9A-Fa-f]{64}"
            r"(?![0-9A-Fa-f])",
            "",
            cleaned,
        )

        cleaned = cleaned.replace(
            path_token,
            master_prompt_path,
        )

        if master_prompt_sha is not None:
            cleaned = cleaned.replace(
                sha_token,
                master_prompt_sha,
            )
            if master_prompt_sha not in cleaned:
                cleaned = cleaned.replace(
                    master_prompt_path,
                    master_prompt_path + " (SHA-256: " + master_prompt_sha + ")",
                    1,
                )

        # A preregistration digest is secondary machine provenance. After
        # full-digest removal, do not leave a misleading empty hash
        # placeholder such as "(manifest SHA-256: )". The preregistration
        # study identifier remains publication-facing; its digest remains
        # available in the archived provenance bundle.
        cleaned = re.sub(
            r"(?i)"
            r"\(\s*"
            r"(?:preregistration\s+)?"
            r"manifest\s+sha\s*-?\s*256"
            r"\s*[:=]\s*"
            r"(?:[0-9A-Fa-f]{64})?"
            r"\s*\)",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)"
            r"\bpreregistration[_\s-]*sha(?:[_\s-]*256)?"
            r"\s*[:=]\s*"
            r"(?:[0-9A-Fa-f]{64})?",
            "",
            cleaned,
        )

        # Disclosure must remain publication-facing rather than an artifact
        # inventory or a second Methods section.
        cleaned = re.sub(
            r"(?i)\s*Key archived artifacts:\s*[^.]*\.",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\s*Generation semantics clarification:\s*[^.]*\.",
            "",
            cleaned,
        )

        # Repair punctuation/whitespace left by removal of secondary hashes.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(
            r"\(\s*\)",
            "",
            cleaned,
        )

        return cleaned

    def sanitize_value(
        value: Any,
        *,
        preserve: bool = False,
    ) -> Any:
        if preserve:
            return value

        if isinstance(value, str):
            return sanitize_text(value)

        if isinstance(value, list):
            return [
                sanitize_value(item)
                for item in value
            ]

        if isinstance(value, dict):
            result: dict[str, Any] = {}

            for key, item in value.items():
                # Citation identifiers are machine identifiers used to build
                # the bibliography and must not be rewritten. Disclosure is
                # sanitized separately so that exactly the mandatory
                # master-prompt locator/digest survives.
                if (
                    key == "disclosure_statement"
                    and isinstance(item, str)
                ):
                    result[key] = sanitize_disclosure_text(
                        item
                    )
                    continue

                preserve_field = key in {
                    "cited_record_ids",
                }

                result[key] = sanitize_value(
                    item,
                    preserve=preserve_field,
                )

            return result

        return value

    if hasattr(manuscript, "model_dump"):
        original_type = type(manuscript)
        sanitized_data = sanitize_value(
            manuscript.model_dump()
        )
        return original_type.model_validate(
            sanitized_data
        )

    if isinstance(manuscript, dict):
        return sanitize_value(manuscript)

    raise TypeError(
        "Structured manuscript sanitizer requires a Pydantic "
        "manuscript model or dictionary."
    )



def audit_manuscript_artifact_references(
    *,
    manuscript: ManuscriptPackage | dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """
    Deterministically verify concrete manuscript provenance claims.

    Hard failures are limited to:
      1. artifacts explicitly claimed to be archived/bundled/provided;
      2. explicit artifact-path + SHA-256 assertions;
      3. structured provider-trace path/hash assertions.

    Mere mention of a path in a reproduction command is not interpreted
    as a claim that the output already exists.
    """
    import glob
    import re

    text = _manuscript_text(manuscript)

    artifact_prefixes = (
        "literature",
        "selection",
        "design",
        "preregistration",
        "execution",
        "analysis",
        "manuscript",
        "disclosure",
    )

    prefix_pattern = "|".join(
        re.escape(x)
        for x in artifact_prefixes
    )

    path_token = (
        rf"(?:{prefix_pattern})/"
        r"[A-Za-z0-9_./*?\-]+"
    )

    sha_token = r"[a-fA-F0-9]{64}"

    issues: list[str] = []
    checked_paths: list[str] = []
    checked_hash_claims: list[dict[str, Any]] = []

    def normalise_path(value: str) -> str:
        value = _normalise_claimed_artifact_path(value)
        return value

    def check_existence(
        relative: str,
        *,
        reason: str,
    ) -> None:
        relative = normalise_path(relative)

        # Directory mentions are not file assertions.
        if relative.endswith("/"):
            return

        checked_paths.append(relative)

        if "*" in relative or "?" in relative:
            if not glob.glob(str(run_dir / relative)):
                issues.append(
                    f"{reason} glob has no matches: {relative}"
                )
            return

        if not (run_dir / relative).is_file():
            issues.append(
                f"{reason} does not exist: {relative}"
            )

    def check_hash(
        relative: str,
        claimed_sha: str,
    ) -> None:
        relative = normalise_path(relative)
        claimed_sha = claimed_sha.lower()

        if relative.endswith("/"):
            return

        if "*" in relative or "?" in relative:
            return

        candidate = run_dir / relative
        actual_sha = (
            sha256_file(candidate)
            if candidate.is_file()
            else None
        )

        checked_hash_claims.append(
            {
                "path": relative,
                "claimed_sha256": claimed_sha,
                "exists": candidate.is_file(),
                "actual_sha256": actual_sha,
            }
        )

        if not candidate.is_file():
            issues.append(
                "Artifact path with explicit SHA-256 "
                f"does not exist: {relative}"
            )
            return

        if actual_sha.lower() != claimed_sha:
            issues.append(
                "Artifact SHA-256 mismatch: "
                f"{relative}; claimed={claimed_sha}; "
                f"actual={actual_sha}"
            )

    # ---------------------------------------------------------
    # A. Explicit archived/provided/stored claims where the
    # provenance verb appears before the path.
    # ---------------------------------------------------------
    before_path_claim = re.compile(
        rf"(?is)"
        rf"(?:"
        rf"archived\s+(?:as|at|under)|"
        rf"stored\s+(?:as|at|under)|"
        rf"available\s+(?:as|at|under)|"
        rf"provided\s+(?:as|at|under)"
        rf")"
        rf"\s+"
        rf"(?P<path>{path_token})"
    )

    for match in before_path_claim.finditer(text):
        check_existence(
            match.group("path"),
            reason="Claimed archived artifact",
        )

    # ---------------------------------------------------------
    # B. Explicit after-path archive/bundle assertions.
    #
    # Only accept wording syntactically attached to the exact
    # artifact path. Do not use generic proximity, because paths
    # appearing as outputs of reproduction commands must not be
    # interpreted as pre-existing archived artifacts.
    # ---------------------------------------------------------
    attached_after_path_claim = re.compile(
        rf"(?is)"
        rf"(?P<path>{path_token})"
        rf"\s*"
        rf"(?:"
        rf"\("
        rf"[^)]{{0,100}}"
        rf"\b(?:archived|bundled|provided|stored)\b"
        rf"[^)]{{0,100}}"
        rf"\)"
        rf"|"
        rf"\s+(?:is|was|are|were)\s+"
        rf"(?:archived|bundled|provided|stored)"
        rf")"
    )

    for match in attached_after_path_claim.finditer(text):
        check_existence(
            match.group("path"),
            reason="Claimed bundled/archived artifact",
        )

    # ---------------------------------------------------------
    # C. Explicit file-path + SHA-256 assertions.
    #
    # Only interpret a digest as belonging to a file when the
    # syntax directly attaches "SHA-256" to that exact path.
    #
    # Valid examples:
    #   analysis/results.json SHA-256 = <digest>
    #   analysis/results.json (SHA-256 = <digest>)
    #
    # Deliberately NOT matched:
    #   analysis/results.json (input results SHA-256 = <digest>)
    #
    # The latter describes a digest stored/mentioned inside the
    # artifact rather than necessarily the artifact's own digest.
    # ---------------------------------------------------------
    explicit_file_hash_claim = re.compile(
        rf"(?is)"
        rf"(?P<path>{path_token})"
        rf"\s*"
        rf"(?:"
        rf"\(\s*"
        rf"|"
        rf"[,;:]?\s*"
        rf")"
        rf"SHA[\-\u2010-\u2015 ]*256"
        rf"\s*(?:=|:)?\s*"
        rf"(?P<sha>{sha_token})"
    )

    seen_hash_pairs: set[tuple[str, str]] = set()

    for match in explicit_file_hash_claim.finditer(text):
        relative = normalise_path(
            match.group("path")
        )

        if (
            relative.endswith("/")
            or "*" in relative
            or "?" in relative
        ):
            continue

        claimed_sha = (
            match.group("sha").lower()
        )

        key = (relative, claimed_sha)

        if key in seen_hash_pairs:
            continue

        seen_hash_pairs.add(key)
        check_hash(relative, claimed_sha)

    # ---------------------------------------------------------
    # D. Structured repair-provider provenance assertions.
    #
    # Treat every repair_provider_trace_path occurrence as the
    # start of one record. Search only within that record (up to
    # the next path occurrence) for its corresponding SHA-256.
    # This avoids one malformed/oddly formatted bullet causing a
    # subsequent repair record to be skipped.
    # ---------------------------------------------------------
    provider_path_pattern = re.compile(
        rf"(?is)"
        rf"repair_provider_trace_path"
        rf"\s*[:=]\s*[\"']?"
        rf"(?P<path>{path_token})"
        rf"[\"']?"
    )

    provider_sha_pattern = re.compile(
        rf"(?is)"
        rf"repair_provider_trace_sha256"
        rf"\s*[:=]\s*[\"']?"
        rf"(?P<sha>{sha_token})"
        rf"[\"']?"
    )

    provider_path_matches = list(
        provider_path_pattern.finditer(text)
    )

    for index, path_match in enumerate(
        provider_path_matches
    ):
        record_end = (
            provider_path_matches[index + 1].start()
            if index + 1 < len(provider_path_matches)
            else min(
                len(text),
                path_match.end() + 700,
            )
        )

        record_text = text[
            path_match.end():record_end
        ]

        sha_match = provider_sha_pattern.search(
            record_text
        )

        if sha_match is None:
            continue

        relative = normalise_path(
            path_match.group("path")
        )
        claimed_sha = (
            sha_match.group("sha").lower()
        )

        key = (relative, claimed_sha)

        if key in seen_hash_pairs:
            continue

        seen_hash_pairs.add(key)
        check_hash(relative, claimed_sha)

    issues = sorted(set(issues))

    return {
        "status": (
            "passed"
            if not issues
            else "failed"
        ),
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "checked_archived_paths": sorted(
            set(checked_paths)
        ),
        "checked_hash_claims": checked_hash_claims,
    }



def audit_manuscript_publication_sanity(
    *,
    run_dir: Path,
    overfull_tolerance_pt: float = 5.0,
) -> dict[str, Any]:
    """Deterministic non-scientific publication-quality audit.

    This audit checks final typesetting and manuscript hygiene only.
    It does not evaluate or alter scientific claims or outcomes.
    """
    import re

    final_dir = run_dir / "manuscript" / "final"
    tex_path = final_dir / "manuscript.tex"
    log_path = final_dir / "manuscript.log"

    issues: list[str] = []
    metrics: dict[str, Any] = {
        "overfull_tolerance_pt": overfull_tolerance_pt,
        "significant_overfull_count": 0,
        "maximum_overfull_pt": 0.0,
        "full_sha256_count": 0,
        "bibliography_environment_count": 0,
        "prebibliography_references_heading_count": 0,
        "duplicate_bibliography_doi_count": 0,
        "inline_doi_label_count": 0,
        "raw_command_count": 0,
        "artifact_path_count": 0,
        "reviewer_meta_phrase_count": 0,
        "undefined_reference_warning_count": 0,
        "missing_character_warning_count": 0,
    }

    tex = (
        tex_path.read_text(encoding="utf-8", errors="replace")
        if tex_path.exists()
        else ""
    )
    log = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )

    if not tex_path.exists():
        issues.append(
            "Final manuscript TeX source is missing."
        )

    if not log_path.exists():
        issues.append(
            "Final manuscript LaTeX compilation log is missing."
        )

    # ---------------------------------------------------------
    # A. Significant column/margin overflow.
    # ---------------------------------------------------------
    overfull_pattern = re.compile(
        r"Overfull \\hbox "
        r"\((?P<pt>[0-9]+(?:\.[0-9]+)?)pt too wide\)"
    )

    overfull_values = [
        float(match.group("pt"))
        for match in overfull_pattern.finditer(log)
    ]

    significant = [
        value
        for value in overfull_values
        if value > overfull_tolerance_pt
    ]

    metrics["significant_overfull_count"] = len(
        significant
    )
    metrics["maximum_overfull_pt"] = (
        max(overfull_values)
        if overfull_values
        else 0.0
    )

    if significant:
        issues.append(
            "Final IEEE manuscript contains "
            f"{len(significant)} Overfull \\\\hbox warnings "
            f"greater than {overfull_tolerance_pt:.1f} pt "
            f"(maximum {max(significant):.3f} pt); "
            "text may cross column or page margins."
        )

    # ---------------------------------------------------------
    # B. Full cryptographic digest pollution.
    #
    # One full SHA-256 is allowed for the immutable master
    # prompt in the mandatory Disclosure Statement. Additional
    # full digests belong in machine-readable provenance.
    # ---------------------------------------------------------
    # Renderer may insert \allowbreak{} inside long digests.
    # Remove presentation-only break commands before auditing hashes.
    hash_scan_tex = re.sub(
        r"\\allowbreak\{\}",
        "",
        tex,
    )

    full_hashes = re.findall(
        r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])",
        hash_scan_tex,
    )

    metrics["full_sha256_count"] = len(full_hashes)

    expected_master_prompt_sha256 = (
        "1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba"
    )

    if len(full_hashes) != 1:
        issues.append(
            "Final manuscript must contain exactly one full 64-character "
            "SHA-256 digest: the immutable master-prompt disclosure digest; "
            f"found {len(full_hashes)}."
        )
    elif full_hashes[0].lower() != expected_master_prompt_sha256:
        issues.append(
            "The sole full SHA-256 in the final manuscript does not match "
            "the immutable master-prompt disclosure digest."
        )

    # ---------------------------------------------------------
    # C. Bibliography uniqueness and duplicate-reference hygiene.
    # ---------------------------------------------------------
    bibliography_environment_count = len(
        re.findall(
            r"\\begin\{thebibliography\}",
            tex,
        )
    )
    metrics["bibliography_environment_count"] = (
        bibliography_environment_count
    )

    if bibliography_environment_count != 1:
        issues.append(
            "Final manuscript must contain exactly one "
            "thebibliography environment; found "
            f"{bibliography_environment_count}."
        )

    body_before_bibliography = tex.split(
        r"\begin{thebibliography}",
        1,
    )[0]

    # Detect a second author-generated References section/heading before
    # the renderer's canonical bibliography. This catches the r80 failure
    # mode where manuscript prose already contained its own References
    # block and the renderer subsequently emitted another bibliography.
    references_heading_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"\\(?:section|section\*|subsection|subsection\*)"
        r"\{\s*references\s*\}"
        r"|references"
        r")\s*$"
    )

    prebibliography_references_headings = (
        references_heading_pattern.findall(
            body_before_bibliography
        )
    )

    metrics["prebibliography_references_heading_count"] = len(
        prebibliography_references_headings
    )

    if prebibliography_references_headings:
        issues.append(
            "Final manuscript contains a separate References heading "
            "before the canonical bibliography; references must appear "
            "exactly once."
        )

    bibliography_text = ""
    if bibliography_environment_count >= 1:
        bibliography_text = tex.split(
            r"\begin{thebibliography}",
            1,
        )[1].split(
            r"\end{thebibliography}",
            1,
        )[0]

    # DOI detection is intentionally confined to bibliography material.
    # Normalize case and common TeX/punctuation wrappers before counting.
    doi_matches = re.findall(
        r"(?i)\b10\.\d{4,9}/[^\s{}]+",
        bibliography_text,
    )

    normalized_dois = [
        doi.lower().rstrip(".,;:)]}")
        for doi in doi_matches
    ]

    duplicate_dois = sorted(
        {
            doi
            for doi in normalized_dois
            if normalized_dois.count(doi) > 1
        }
    )

    metrics["duplicate_bibliography_doi_count"] = len(
        duplicate_dois
    )

    if duplicate_dois:
        issues.append(
            "Final bibliography contains duplicate DOI entries: "
            + ", ".join(duplicate_dois)
        )

    # ---------------------------------------------------------
    # D. DOI labels dumped into prose.
    #
    # DOI bibliography fields are fine; literal 'DOI:' labels
    # in manuscript prose are publication metadata pollution.
    # ---------------------------------------------------------
    body_before_bibliography = tex.split(
        r"\begin{thebibliography}",
        1,
    )[0]

    inline_doi_labels = re.findall(
        r"(?i)\bDOI\s*:",
        body_before_bibliography,
    )

    metrics["inline_doi_label_count"] = len(
        inline_doi_labels
    )

    if inline_doi_labels:
        issues.append(
            "Final manuscript contains "
            f"{len(inline_doi_labels)} inline DOI: label(s) "
            "before the bibliography; cite literature normally "
            "and keep DOI metadata in references."
        )

    # Raw DOI strings enclosed in brackets are not IEEE citation
    # markers, e.g. [10.1002/nem.2313]. Scientific prose must use
    # numbered bibliography citations such as [1].
    bracketed_doi_citations = re.findall(
        r"\[\s*10\.\d{4,9}/[^\]\s]+\s*\]",
        body_before_bibliography,
        flags=re.IGNORECASE,
    )

    metrics["bracketed_doi_citation_count"] = len(
        bracketed_doi_citations
    )

    if bracketed_doi_citations:
        issues.append(
            "Final manuscript contains "
            f"{len(bracketed_doi_citations)} raw bracketed DOI "
            "citation(s) in scientific prose; use standard IEEE "
            "numbered citations and keep DOI metadata only in "
            "the bibliography."
        )

    # ---------------------------------------------------------
    # D. Raw command / artifact-path pollution.
    #
    # Reproducibility infrastructure belongs in the archived
    # run, not as shell-command or file-inventory prose in the
    # scientific manuscript.
    # ---------------------------------------------------------
    # Detect command signatures conservatively. We do not
    # require the entire command to be on one unmodified TeX
    # source line because rendering may escape punctuation or
    # insert formatting commands.
    raw_command_patterns = [
        r"(?i)\bpython(?:3)?\s+(?:execution|analysis|provenance|manuscript)/",
        r"(?i)\bjq\s+-[A-Za-z]",
        r"(?i)\bsha256sum\s+",
        r"(?i)\bgrep\s+-?[A-Za-z]",
        r"(?i)\bcurl\s+-?[A-Za-z]",
    ]

    raw_command_hits: list[str] = []

    for pattern in raw_command_patterns:
        raw_command_hits.extend(
            match.group(0)
            for match in re.finditer(
                pattern,
                body_before_bibliography,
            )
        )

    metrics["raw_command_count"] = len(
        raw_command_hits
    )

    if raw_command_hits:
        issues.append(
            "Final manuscript contains raw reproduction "
            f"command(s) in scientific prose: "
            f"{len(raw_command_hits)} occurrence(s)."
        )

    artifact_path_pattern = re.compile(
        r"(?i)\b(?:execution|analysis|provenance|manuscript)"
        r"/[A-Za-z0-9_.*/-]+"
    )

    artifact_paths = artifact_path_pattern.findall(
        body_before_bibliography
    )

    metrics["artifact_path_count"] = len(
        artifact_paths
    )

    # A few concise artifact references are preferable, but path
    # density alone is a publication-style diagnostic rather than a
    # deterministic validity failure. Concrete artifact-path correctness
    # is enforced independently by audit_manuscript_artifact_references().
    # Keep the count above for audit/reporting, but do not reject an
    # otherwise valid manuscript solely because it contains >8 valid
    # run-relative artifact references.

    # ---------------------------------------------------------
    # E. Reviewer-response / future-review meta-language.
    # ---------------------------------------------------------
    reviewer_meta_patterns = [
        r"(?i)\breviewers?\s+requested\b",
        r"(?i)\bif\s+reviewers?\s+(?:require|request|insist)\b",
        r"(?i)\bwe\s+will\s+(?:insert|add|provide)\b",
        r"(?i)\berratum[- ]style\b",
        r"(?i)\bbrief\s+addendum\b",
    ]

    reviewer_meta_hits: list[str] = []

    for pattern in reviewer_meta_patterns:
        reviewer_meta_hits.extend(
            match.group(0)
            for match in re.finditer(pattern, tex)
        )

    metrics["reviewer_meta_phrase_count"] = len(
        reviewer_meta_hits
    )

    if reviewer_meta_hits:
        issues.append(
            "Final manuscript contains reviewer-response or "
            "future-review meta-language: "
            + ", ".join(
                sorted(set(reviewer_meta_hits))
            )
        )

    # ---------------------------------------------------------
    # F. Undefined citations/references in the final LaTeX log.
    # ---------------------------------------------------------
    undefined_reference_patterns = [
        r"(?i)LaTeX Warning: Citation .* undefined",
        r"(?i)LaTeX Warning: Reference .* undefined",
        r"(?i)There were undefined references",
        r"(?i)There were undefined citations",
    ]

    undefined_reference_hits: list[str] = []

    for pattern in undefined_reference_patterns:
        undefined_reference_hits.extend(
            match.group(0)
            for match in re.finditer(
                pattern,
                log,
            )
        )

    metrics["undefined_reference_warning_count"] = len(
        undefined_reference_hits
    )

    if undefined_reference_hits:
        issues.append(
            "Final manuscript compilation contains "
            f"{len(undefined_reference_hits)} undefined "
            "citation/reference warning(s)."
        )

    # ---------------------------------------------------------
    # G. Missing glyphs / characters in the final LaTeX log.
    # ---------------------------------------------------------
    missing_character_hits = re.findall(
        r"(?im)^Missing character:.*$",
        log,
    )

    metrics["missing_character_warning_count"] = len(
        missing_character_hits
    )

    if missing_character_hits:
        issues.append(
            "Final manuscript compilation contains "
            f"{len(missing_character_hits)} missing-character "
            "warning(s); one or more manuscript glyphs may not "
            "render correctly."
        )

    issues = sorted(set(issues))

    return {
        "status": "passed" if not issues else "failed",
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "metrics": metrics,
        "tex_path": str(tex_path),
        "log_path": str(log_path),
    }



def _compact_execution_manifest_for_manuscript(
    execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic manuscript-facing projection of the execution
    manifest.

    The complete manifest remains archived and authoritative. The
    manuscript agents do not need hundreds of kilobytes of per-file
    SHA-256 entries on every revision call.
    """
    compact = {
        key: value
        for key, value in execution_manifest.items()
        if key != "artifact_hashes"
    }

    artifact_hashes = execution_manifest.get(
        "artifact_hashes",
        {},
    )

    if isinstance(artifact_hashes, dict):
        compact["artifact_hash_count"] = len(
            artifact_hashes
        )

        # Preserve a deterministic small set of scientifically useful
        # provenance hashes while avoiding the complete per-file hash
        # inventory in the model context.
        priority_fragments = (
            "execution_manifest",
            "task_manifest",
            "model_configuration",
            "results",
            "analysis",
            "preregistration",
        )

        selected_hashes: dict[str, Any] = {}

        for path in sorted(artifact_hashes):
            if any(
                fragment in path
                for fragment in priority_fragments
            ):
                selected_hashes[path] = artifact_hashes[path]

            if len(selected_hashes) >= 40:
                break

        compact["representative_artifact_hashes"] = (
            selected_hashes
        )

    return compact


def _compact_verified_records_for_manuscript(
    records: list[Any],
) -> list[dict[str, Any]]:
    """
    Deterministic bibliographic projection for manuscript revision.

    Full literature records and abstracts remain archived. Revision
    agents receive citation identity/provenance fields sufficient to
    preserve and verify already-written references without repeatedly
    injecting the full retrieved abstracts.
    """
    compact_records: list[dict[str, Any]] = []

    allowed_fields = (
        "record_id",
        "title",
        "publication_year",
        "doi",
        "url",
        "source_api",
        "authors",
        "cited_by_count",
    )

    for record in records:
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        elif isinstance(record, dict):
            data = record
        else:
            continue

        compact_records.append(
            {
                key: data.get(key)
                for key in allowed_fields
                if key in data
            }
        )

    return compact_records


def _compact_manuscript_evidence_bundle(
    manuscript_evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic bounded projection of manuscript evidence.

    Full artifact examples and representative task collections remain
    archived. A bounded deterministic sample is sufficient for revision
    and review while preventing context growth from scaling with the
    complete experiment.
    """
    compact = dict(manuscript_evidence_bundle)

    for key, limit in (
        ("artifact_examples", 12),
        ("representative_tasks", 8),
    ):
        value = compact.get(key)

        if isinstance(value, list):
            compact[key] = value[:limit]
            compact[f"{key}_total_count"] = len(value)
            compact[f"{key}_context_count"] = min(
                len(value),
                limit,
            )

    return compact



def _compact_evidence_synthesis_for_manuscript(
    synthesis: Any,
    *,
    max_list_items: int = 12,
    max_string_chars: int = 5000,
    max_total_chars: int = 50000,
) -> Any:
    """
    Deterministically compact the autonomous literature synthesis for
    manuscript drafting/revision.

    The purpose is to keep scientific synthesis readily available to the
    manuscript agents without reintroducing the large-context failure that
    motivated manuscript context compaction.

    This preserves structure and leading substantive content while bounding
    large lists and long strings. It does not generate, reinterpret, or add
    scientific content.
    """

    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): compact(child)
                for key, child in value.items()
            }

        if isinstance(value, list):
            bounded = [
                compact(child)
                for child in value[:max_list_items]
            ]

            if len(value) > max_list_items:
                bounded.append(
                    {
                        "_context_truncation": True,
                        "_total_items": len(value),
                        "_included_items": max_list_items,
                    }
                )

            return bounded

        if isinstance(value, str):
            if len(value) <= max_string_chars:
                return value

            return (
                value[:max_string_chars]
                + "\n[deterministically truncated for manuscript context]"
            )

        return value

    compacted = compact(synthesis)

    encoded = json.dumps(
        compacted,
        ensure_ascii=False,
        sort_keys=True,
    )

    if len(encoded) <= max_total_chars:
        return compacted

    # Second deterministic tightening pass if a structurally large synthesis
    # still exceeds the manuscript-context budget.
    def tighten(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): tighten(child)
                for key, child in value.items()
            }

        if isinstance(value, list):
            bounded = [
                tighten(child)
                for child in value[:6]
            ]

            if len(value) > 6:
                bounded.append(
                    {
                        "_context_truncation": True,
                        "_total_items": len(value),
                        "_included_items": 6,
                    }
                )

            return bounded

        if isinstance(value, str):
            if len(value) <= 2500:
                return value

            return (
                value[:2500]
                + "\n[deterministically truncated for manuscript context]"
            )

        return value

    tightened = tighten(synthesis)

    return {
        "scientific_literature_synthesis": tightened,
        "_context_note": (
            "Deterministically bounded projection of the autonomous "
            "literature evidence synthesis. The complete synthesis remains "
            "archived in literature/evidence_synthesis.json."
        ),
    }




def _compact_execution_manifest_for_analysis_planning(
    execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic bounded projection of the completed execution manifest
    for ANALYSIS_PLANNER.

    Preserve all execution-semantic and accounting fields, but omit the
    potentially very large artifact_hashes inventory. The complete manifest
    remains archived and authoritative for deterministic validation.
    """
    compact = dict(execution_manifest)

    artifact_hashes = compact.pop(
        "artifact_hashes",
        None,
    )

    if isinstance(artifact_hashes, dict):
        compact["artifact_hashes_total_count"] = len(
            artifact_hashes
        )

    return compact



CONVENTIONAL_MANUSCRIPT_TITLE_INSTRUCTION = (
    "Use a conventional academic research-paper title appropriate for "
    "IEEE/CNSM. Do not use arrow chains, workflow notation, pipeline "
    "notation, slogans, or slide-style titles. In particular, do not use "
    "sequences such as 'A → B → C' or 'A -> B -> C' in the title. "
    "Prefer a concise descriptive title or, where scientifically "
    "appropriate, a question-based title. A conventional title with an "
    "optional colon is preferred."
)


def _manuscript_revision_context(
    *,
    records: list[Any],
    execution_manifest: dict[str, Any],
    manuscript_evidence_bundle: dict[str, Any],
    evidence_synthesis: Any,
) -> dict[str, Any]:
    """
    Build the bounded deterministic context shared by all manuscript
    revision calls.
    """
    return {
        "verified_records": (
            _compact_verified_records_for_manuscript(
                records
            )
        ),
        "execution_manifest": (
            _compact_execution_manifest_for_manuscript(
                execution_manifest
            )
        ),
        "manuscript_evidence_bundle": (
            _compact_manuscript_evidence_bundle(
                manuscript_evidence_bundle
            )
        ),
        "evidence_synthesis": (
            _compact_evidence_synthesis_for_manuscript(
                evidence_synthesis
            )
        ),
    }


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

def compact_verified_records_for_manuscript(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deterministically compact verified literature records for
    manuscript generation.

    Preserve bibliographic identity and concise evidence needed for
    citation/related-work writing while excluding verbose provider
    metadata that is not required by the manuscript author.
    """
    retained_fields = (
        "id",
        "title",
        "authors",
        "author",
        "year",
        "publication_year",
        "venue",
        "journal",
        "publisher",
        "doi",
        "url",
        "abstract",
        "source",
        "verified",
        "verification_status",
    )

    compact_records: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        compact: dict[str, Any] = {}

        for field in retained_fields:
            if field not in record:
                continue

            value = record[field]

            # Abstracts are useful scientifically but can dominate the
            # manuscript-author context. Retain a deterministic prefix.
            if (
                field == "abstract"
                and isinstance(value, str)
            ):
                value = value[:1200]

            compact[field] = value

        compact_records.append(compact)

    return compact_records

def compact_execution_manifest_for_manuscript(
    execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministically compact the execution manifest for manuscript
    generation while preserving execution semantics and provenance summary.
    """

    compact = dict(execution_manifest)

    artifact_hashes = compact.pop(
        "artifact_hashes",
        {},
    )

    compact["artifact_hash_summary"] = {
        "artifact_count": (
            len(artifact_hashes)
            if isinstance(
                artifact_hashes,
                dict,
            )
            else 0
        ),
        "full_hash_manifest_available": True,
        "full_hash_manifest_path": (
            "execution/execution_manifest.json"
        ),
    }

    return compact

def build_manuscript_evidence_bundle(
    run_dir: Path,
    execution_manifest: dict[str, Any],
    analysis_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact, deterministic evidence bundle for manuscript revision.

    The bundle is derived only from existing autonomous-run artifacts.
    It does not create new scientific results or alter the analysis.
    """
    bundle: dict[str, Any] = {
        "execution_summary": (
            compact_execution_manifest_for_manuscript(
                execution_manifest
            )
        ),
        "analysis_results": analysis_results,
        "artifact_examples": [],
        "analysis_artifacts": {},
    }

    scoring_dir = (
        run_dir
        / "execution"
        / "scoring"
    )

    if scoring_dir.is_dir():
        scoring_files = sorted(
            scoring_dir.glob("*.json")
        )

        # Deterministic representative sample:
        # first three baseline and first three guarded files.
        baseline_files = [
            p for p in scoring_files
            if p.name.endswith("-baseline.json")
        ][:3]

        guarded_files = [
            p for p in scoring_files
            if p.name.endswith("-guarded.json")
        ][:3]

        for p in baseline_files + guarded_files:
            data = read_json(p)

            bundle["artifact_examples"].append(
                {
                    "path": str(
                        p.relative_to(run_dir)
                    ),
                    "sha256": sha256_file(p),
                    "content": data,
                }
            )

    analysis_dir = run_dir / "analysis"

    for name in (
        "condition_summary.csv",
        "paired_contingency_table.csv",
        "contamination_summary.csv",
        "missingness_summary.csv",
        "analysis_log.jsonl",
        "deterministic_reconciliation.json",
    ):
        p = analysis_dir / name

        if not p.is_file():
            continue

        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )

        bundle["analysis_artifacts"][name] = {
            "path": str(
                p.relative_to(run_dir)
            ),
            "sha256": sha256_file(p),
            "content": text[:12000],
        }

    model_configuration_path = (
        run_dir
        / "execution"
        / "model_configuration.json"
    )

    if model_configuration_path.is_file():
        bundle["model_configuration"] = {
            "path": str(
                model_configuration_path.relative_to(
                    run_dir
                )
            ),
            "sha256": sha256_file(
                model_configuration_path
            ),
            "content": read_json(
                model_configuration_path
            ),
        }

    master_prompt_path = (
        run_dir
        / "provenance"
        / "master_prompt.txt"
    )

    master_prompt_hash_path = (
        run_dir
        / "provenance"
        / "master_prompt.sha256"
    )

    if (
        master_prompt_path.is_file()
        and master_prompt_hash_path.is_file()
    ):
        bundle["initial_master_prompt_reference"] = {
            "path": str(
                master_prompt_path.relative_to(
                    run_dir
                )
            ),
            "sha256": (
                master_prompt_hash_path
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            ),
        }

    task_manifest_path = (
        run_dir
        / "execution"
        / "task_manifest.jsonl"
    )

    bundle["representative_tasks"] = []

    if task_manifest_path.is_file():
        task_rows: list[dict[str, Any]] = []

        with task_manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)

                if isinstance(row, dict):
                    task_rows.append(row)

        # Deterministic diversity sample based on the task's
        # archived difficulty pattern. No outcome-based selection.
        seen_patterns: set[str] = set()

        for row in task_rows:
            task_payload = row.get(
                "task_payload",
                {},
            )

            difficulty = (
                task_payload.get(
                    "difficulty",
                    {},
                )
                if isinstance(
                    task_payload,
                    dict,
                )
                else {}
            )

            pattern = difficulty.get(
                "pattern",
                "unknown",
            )

            if pattern in seen_patterns:
                continue

            seen_patterns.add(pattern)

            task_id = row.get("task_id")

            representative = {
                "task_id": task_id,
                "task_manifest_entry": row,
            }

            if isinstance(task_id, str):
                response_entries = {}

                for condition in (
                    "shared-initial",
                    "baseline",
                    "guarded",
                ):
                    response_path = (
                        run_dir
                        / "execution"
                        / "responses"
                        / f"{task_id}-{condition}.txt"
                    )

                    if not response_path.is_file():
                        continue

                    response_entries[
                        condition
                    ] = {
                        "path": str(
                            response_path.relative_to(
                                run_dir
                            )
                        ),
                        "sha256": sha256_file(
                            response_path
                        ),
                        "content": (
                            response_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )[:2000]
                        ),
                    }

                representative[
                    "responses"
                ] = response_entries

            bundle[
                "representative_tasks"
            ].append(
                representative
            )

            if (
                len(
                    bundle[
                        "representative_tasks"
                    ]
                )
                >= 6
            ):
                break

    return bundle


def manuscript_section_word_count(
    manuscript: ManuscriptPackage,
) -> int:
    """Deterministically count words in manuscript scientific sections."""
    payload = manuscript.model_dump()
    sections = payload.get("sections", {})

    if not isinstance(sections, dict):
        return 0

    return sum(
        len(str(value).split())
        for value in sections.values()
    )

def preregistration_analysis_contract_issues(
    preregistration: PreregistrationDocument,
    *,
    analysis_contracts: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate the sealed primary estimand against registered analysis."""
    issues: list[str] = []

    estimand_id = preregistration.primary_estimand_id

    matching = [
        family
        for family, contract
        in analysis_contracts.items()
        if contract.get("estimand") == estimand_id
    ]

    if not matching:
        issues.append(
            "Preregistration primary_estimand_id is not "
            "supported by any registered analysis executor."
        )

    return sorted(set(issues))

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

            error_text = str(exc).lower()

            if (
                "context_length_exceeded"
                in error_text
                or "exceeds the context window"
                in error_text
            ):
                raise RuntimeError(
                    f"{stage_name} exceeded the model "
                    "context window; identical retries "
                    "are prohibited."
                ) from exc

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

def required_confirmatory_task_count(
    repaired_design: RepairedStudyDesign,
) -> int:
    """Resolve the autonomous design's selected confirmatory sample size."""
    recommended_id = (
        repaired_design.power_plan.recommended_scenario_id
    )

    matches = [
        scenario
        for scenario in repaired_design.budget_scenarios
        if scenario.scenario_id == recommended_id
    ]

    if len(matches) != 1:
        raise ValueError(
            "Exactly one budget scenario must match "
            "power_plan.recommended_scenario_id."
        )

    return int(matches[0].confirmatory_items)



def preregistration_scientific_coherence_issues(
    preregistration: Any,
) -> list[str]:
    """
    Deterministic design-time coherence checks only.

    Does not inspect execution outcomes.
    """
    hypotheses = " ".join(
        str(x)
        for x in preregistration.confirmatory_hypotheses
    ).lower()

    estimand = (
        str(preregistration.primary_estimand)
        + " "
        + str(preregistration.primary_estimand_id)
    ).lower()

    analysis = str(
        preregistration.analysis_plan
    ).lower()

    issues: list[str] = []

    predictive_claim = any(
        token in hypotheses
        for token in (
            "predict",
            "explain",
            "variance",
            "correlat",
            "association",
            "discriminat",
        )
    )

    paired_difference_estimand = (
        "paired_success_rate_difference" in estimand
        or (
            "guarded" in estimand
            and "baseline" in estimand
            and "difference" in estimand
        )
    )

    predictive_analysis = any(
        token in analysis
        for token in (
            "regression",
            "r-squared",
            "r²",
            "auc",
            "predictive",
            "correlation",
            "association",
        )
    )

    if (
        predictive_claim
        and paired_difference_estimand
        and not predictive_analysis
    ):
        issues.append(
            "Confirmatory hypothesis makes a predictive/explanatory "
            "comparison, but the primary estimand/test is a paired "
            "guarded-versus-baseline success difference. Rewrite the "
            "hypothesis or choose an estimand and analysis that directly "
            "test the same scientific claim."
        )

    return issues


def preregistration_identifiability_issues(
    preregistration: Any,
) -> list[str]:
    """
    Detect only structural non-identifiability from the preregistered
    execution contract.

    Do not require observed failures, injected faults, or multiple draws:
    naturally occurring generator failures can make a paired repair
    estimand identifiable under shared-initial-candidate semantics.
    """
    contract = preregistration.execution_contract.model_dump()

    estimand = (
        str(preregistration.primary_estimand)
        + " "
        + str(preregistration.primary_estimand_id)
    ).lower()

    repair_difference = (
        "paired_success_rate_difference" in estimand
        or (
            "guarded" in estimand
            and "baseline" in estimand
            and "difference" in estimand
        )
    )

    if not repair_difference:
        return []

    conditions = {
        str(value).strip().lower()
        for value in contract.get("conditions", [])
    }

    maximum_repair_calls = int(
        contract.get("maximum_repair_calls_per_task") or 0
    )

    issues: list[str] = []

    if not {"baseline", "guarded"}.issubset(conditions):
        issues.append(
            "Primary guarded-versus-baseline estimand requires both "
            "baseline and guarded execution conditions."
        )

    if maximum_repair_calls < 1:
        issues.append(
            "Primary guarded-versus-baseline repair estimand is "
            "structurally non-identifiable because the execution "
            "contract permits no repair call in the guarded condition."
        )

    return issues



def preregistration_execution_contract_issues(
    preregistration: PreregistrationDocument,
    *,
    planning_contracts: dict[str, dict[str, Any]],
    available_execution_models: list[str],
    required_task_count: int,
) -> list[str]:
    """
    Verify that structured preregistration fields and scientific prose
    describe an experiment actually executable by the selected adapter.
    """
    issues: list[str] = []
    declared = preregistration.execution_contract.model_dump()
    adapter_family = str(declared["adapter_family"])
    adapter_contract = planning_contracts.get(adapter_family)

    if adapter_contract is None:
        return [
            "Preregistration selected an unregistered "
            f"adapter_family: {adapter_family}"
        ]

    for field in (
        "execution_mode",
        "design",
        "conditions",
        "model_provider",
        "generation_semantics",
        "independent_condition_generation",
        "initial_generation_calls_per_task",
        "maximum_repair_calls_per_task",
        "retrieval_augmented_generation",
    ):
        if declared.get(field) != adapter_contract.get(field):
            issues.append(
                "Preregistration execution_contract "
                f"{field} does not match the registered "
                "adapter planning contract."
            )

    if declared["model_names"] != available_execution_models:
        issues.append(
            "Preregistration model_names must exactly match the frozen "
            "available execution models."
        )

    if preregistration.model_scope != declared["model_names"]:
        issues.append(
            "Preregistration model_scope must exactly match "
            "execution_contract.model_names."
        )

    expected_transformations = list(
        adapter_contract.get("transformations", {}).values()
    )
    if preregistration.transformation_scope != expected_transformations:
        issues.append(
            "Preregistration transformation_scope must exactly match the "
            "registered adapter transformations."
        )

    prereg_text = " ".join(
        [
            preregistration.title,
            preregistration.research_question,
            preregistration.primary_estimand,
            preregistration.sampling_plan,
            preregistration.analysis_plan,
            *preregistration.confirmatory_hypotheses,
            *preregistration.benchmark_scope,
            *preregistration.transformation_scope,
        ]
    ).lower()

    # The paired adapter guarantees reuse of one initial candidate across
    # conditions. That does not imply deterministic sampling by the hosted
    # model. Preregistration must not invent stronger sampling guarantees
    # than the registered adapter provides.
    deterministic_sampling_claims = (
        "temperature=0",
        "temperature = 0",
        "temperature of 0",
        "zero temperature",
        "deterministic initial llm output",
        "deterministic initial model output",
        "deterministic llm generation",
        "deterministic model generation",
        "deterministic initial generation",
        "sampling variance eliminated",
        "eliminate sampling variance",
    )

    if (
        not adapter_contract.get(
            "guarantees_deterministic_model_sampling",
            False,
        )
        and any(
            phrase in prereg_text
            for phrase in deterministic_sampling_claims
        )
    ):
        issues.append(
            "Preregistration claims deterministic hosted-model sampling "
            "(for example temperature=0 or deterministic initial model "
            "generation), but the selected adapter contract guarantees "
            "only shared_initial_candidate pairing and does not guarantee "
            "deterministic model sampling. Preregister one initial model "
            "generation per task shared unchanged across conditions and "
            "report the runtime sampling parameters after execution."
        )

    # A textual promise of a fixed randomized holdout seed is invalid unless
    # that seed is actually represented by the registered execution contract.
    if (
        "holdout" in prereg_text
        and "fixed seed" in prereg_text
        and adapter_contract.get("holdout_selection_seed") is None
    ):
        issues.append(
            "Preregistration claims a fixed holdout-selection seed, but "
            "the selected adapter contract contains no machine-readable "
            "holdout_selection_seed. Remove the unsupported fixed-seed "
            "claim or use an execution contract that explicitly freezes "
            "and records that seed."
        )

    if (
        adapter_contract.get("supports_multi_model_consensus", False) is False
        and any(phrase in prereg_text for phrase in (
            "multi-model consensus", "multimodel consensus",
            "multi model consensus", "3-model consensus",
            "3 model consensus", "three-model consensus",
            "three model consensus", "model ensemble",
            "multi-model ensemble", "multimodel ensemble",
            "consensus ensemble", "ensemble defense",
            "ensemble defence", "majority vote across models",
            "majority voting across models",
        ))
    ):
        issues.append(
            "Preregistration describes a multi-model consensus or ensemble "
            "execution stage, but the selected registered adapter does not "
            "execute multi-model consensus."
        )

    if (
        adapter_contract.get("supports_simulated_human_gate", False) is False
        and any(phrase in prereg_text for phrase in (
            "simulated human gate", "simulated human review",
            "human gate", "human-in-the-loop gate",
        ))
    ):
        issues.append(
            "Preregistration describes a simulated-human or human-gate stage, "
            "but the selected registered adapter does not execute such a stage."
        )

    if (
        adapter_contract.get("supports_prompt_family_stratification", False) is False
        and all(label in prereg_text for label in (
            "benign", "ambiguous", "adversarial",
        ))
    ):
        issues.append(
            "Preregistration describes benign/ambiguous/adversarial prompt-family "
            "stratification, but the selected registered adapter does not "
            "generate or record those strata."
        )

    if (
        adapter_contract.get("retrieval_augmented_generation") is False
        and any(phrase in prereg_text for phrase in (
            "retrieval-augmented", "retrieval augmented", "rag+", "rag +",
        ))
    ):
        issues.append(
            "Preregistration describes retrieval-augmented generation, but the "
            "registered adapter does not execute retrieval-augmented generation."
        )

    if (
        adapter_contract.get("independent_condition_generation") is False
        and any(phrase in prereg_text for phrase in (
            "independent generation", "independently generated",
            "separate generation", "separately generated",
        ))
    ):
        issues.append(
            "Preregistration describes independent per-condition generation, "
            "but the registered adapter uses one shared initial candidate."
        )

    if int(declared["task_count"]) != required_task_count:
        issues.append(
            "Preregistration task_count must equal required_confirmatory_task_count."
        )

    expected_episode_count = required_task_count * int(
        adapter_contract["episodes_per_task"]
    )
    if int(declared["planned_episode_count"]) != expected_episode_count:
        issues.append(
            "Preregistration planned_episode_count does not match the executable "
            "adapter contract."
        )

    expected_maximum_model_calls = required_task_count * int(
        adapter_contract["maximum_model_calls_per_task"]
    )
    if int(declared["maximum_model_calls"]) != expected_maximum_model_calls:
        issues.append(
            "Preregistration maximum_model_calls does not match the executable "
            "adapter contract."
        )

    return sorted(set(issues))

def canonicalize_preregistration_analysis_contract(
    preregistration: PreregistrationDocument,
    *,
    analysis_contracts: dict[str, dict[str, Any]],
) -> PreregistrationDocument:
    """
    Canonicalize the machine-readable primary estimand identifier
    only when exactly one registered analysis contract is compatible
    with the already-selected execution adapter.

    This is foreign-key normalization, not scientific selection:
    hypotheses, the human-readable primary estimand, analysis prose,
    outcomes, and scientific interpretation are left unchanged.
    """
    adapter_family = (
        preregistration.execution_contract.adapter_family
    )

    compatible_contracts = [
        contract
        for contract in analysis_contracts.values()
        if adapter_family
        in contract.get(
            "compatible_execution_adapter_families",
            [],
        )
    ]

    # Never choose deterministically if multiple scientifically
    # executable analysis contracts remain available.
    if len(compatible_contracts) != 1:
        return preregistration

    canonical_estimand_id = compatible_contracts[0].get(
        "estimand"
    )

    if (
        not isinstance(canonical_estimand_id, str)
        or not canonical_estimand_id.strip()
    ):
        return preregistration

    preregistration.primary_estimand_id = (
        canonical_estimand_id
    )

    return preregistration


def canonicalize_preregistration_execution_contract(
    preregistration: PreregistrationDocument,
    *,
    planning_contracts: dict[str, dict[str, Any]],
    available_execution_models: list[str],
    required_task_count: int,
) -> PreregistrationDocument:
    """
    Canonicalize fields that are mechanically fixed by the selected
    registered execution adapter.

    This does not alter hypotheses, estimands, analysis choices,
    benchmark scope, or other scientific content.
    """
    adapter_family = (
        preregistration.execution_contract.adapter_family
    )

    adapter_contract = planning_contracts.get(
        adapter_family
    )

    if adapter_contract is None:
        return preregistration

    episodes_per_task = int(
        adapter_contract["episodes_per_task"]
    )

    maximum_model_calls_per_task = int(
        adapter_contract[
            "maximum_model_calls_per_task"
        ]
    )

    preregistration.execution_contract.execution_mode = (
        adapter_contract["execution_mode"]
    )
    preregistration.execution_contract.design = (
        adapter_contract["design"]
    )
    preregistration.execution_contract.conditions = list(
        adapter_contract["conditions"]
    )
    preregistration.execution_contract.model_provider = (
        adapter_contract["model_provider"]
    )

    preregistration.execution_contract.model_names = list(
        available_execution_models
    )

    preregistration.execution_contract.task_count = (
        required_task_count
    )

    preregistration.execution_contract.planned_episode_count = (
        required_task_count * episodes_per_task
    )

    preregistration.execution_contract.maximum_model_calls = (
        required_task_count
        * maximum_model_calls_per_task
    )

    preregistration.execution_contract.generation_semantics = (
        adapter_contract["generation_semantics"]
    )

    preregistration.execution_contract.independent_condition_generation = (
        adapter_contract[
            "independent_condition_generation"
        ]
    )

    preregistration.execution_contract.initial_generation_calls_per_task = (
        int(
            adapter_contract[
                "initial_generation_calls_per_task"
            ]
        )
    )

    preregistration.execution_contract.maximum_repair_calls_per_task = (
        int(
            adapter_contract[
                "maximum_repair_calls_per_task"
            ]
        )
    )

    preregistration.execution_contract.retrieval_augmented_generation = (
        bool(
            adapter_contract[
                "retrieval_augmented_generation"
            ]
        )
    )

    preregistration.model_scope = list(
        available_execution_models
    )

    preregistration.transformation_scope = list(
        adapter_contract[
            "transformations"
        ].values()
    )

    return preregistration

def build_deterministic_reconciliation(
    *,
    experiment_plan: dict[str, Any],
    execution_manifest: dict[str, Any],
    analysis_results: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    confirmatory_results = (
        analysis_results.get(
            "confirmatory_results",
            [],
        )
    )

    if len(confirmatory_results) != 1:
        raise ValueError(
            "Deterministic reconciliation requires "
            "exactly one confirmatory result."
        )

    result = confirmatory_results[0]

    n_11 = int(result["n_11"])
    n_10 = int(result["n_10"])
    n_01 = int(result["n_01"])
    n_00 = int(result["n_00"])

    contingency_total = (
        n_11 + n_10 + n_01 + n_00
    )

    baseline_from_contingency = (
        n_11 + n_01
    )

    guarded_from_contingency = (
        n_11 + n_10
    )

    complete_pair_count = int(
        result["complete_pair_count"]
    )

    baseline_success_count = int(
        result["baseline_success_count"]
    )

    guarded_success_count = int(
        result["guarded_success_count"]
    )

    missingness = dict(
        analysis_results.get(
            "missingness_summary",
            {},
        )
    )

    pair_slots_from_missingness = sum(
        int(
            missingness.get(
                key,
                0,
            )
        )
        for key in (
            "complete_pairs",
            "baseline_only_observed_pairs",
            "guarded_only_observed_pairs",
            "both_missing_pairs",
        )
    )

    task_count = int(
        experiment_plan["task_count"]
    )

    conditions = list(
        experiment_plan["conditions"]
    )

    planned_episode_count = int(
        execution_manifest[
            "planned_episode_count"
        ]
    )

    completed_episode_count = int(
        execution_manifest[
            "completed_episode_count"
        ]
    )

    failed_episode_count = int(
        execution_manifest[
            "failed_episode_count"
        ]
    )

    execution_log_path = (
        run_dir
        / str(
            execution_manifest[
                "execution_log_path"
            ]
        )
    )

    cache_hit_count = 0
    cache_miss_count = 0
    hosted_model_call_event_count = 0
    completed_hosted_model_call_event_count = 0
    failed_hosted_model_call_event_count = 0

    stage_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}

    cache_key_conditions: dict[
        str,
        set[str],
    ] = {}

    if execution_log_path.is_file():
        for line in execution_log_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue

            event = json.loads(line)

            if (
                event.get("event_type")
                != "hosted_model_call"
            ):
                continue

            hosted_model_call_event_count += 1

            outcome = str(
                event.get(
                    "outcome",
                    "",
                )
            )

            if outcome == "COMPLETED":
                completed_hosted_model_call_event_count += 1
            elif outcome == "FAILED":
                failed_hosted_model_call_event_count += 1

            stage = str(
                event.get(
                    "stage",
                    "",
                )
            )

            condition = str(
                event.get(
                    "condition",
                    "",
                )
            )

            if stage:
                stage_counts[stage] = (
                    stage_counts.get(
                        stage,
                        0,
                    )
                    + 1
                )

            if condition:
                condition_counts[
                    condition
                ] = (
                    condition_counts.get(
                        condition,
                        0,
                    )
                    + 1
                )

            cache_status = event.get(
                "cache_status"
            )

            if cache_status == "HIT":
                cache_hit_count += 1
            elif cache_status == "MISS":
                cache_miss_count += 1

            cache_key = event.get(
                "cache_key_sha256"
            )

            if cache_key and condition:
                cache_key_conditions.setdefault(
                    str(cache_key),
                    set(),
                ).add(condition)

    cross_condition_cache_keys = sorted(
        cache_key
        for cache_key, seen_conditions
        in cache_key_conditions.items()
        if len(seen_conditions) > 1
    )

    marginal_contingency_consistent = (
        contingency_total
        == complete_pair_count
        and baseline_from_contingency
        == baseline_success_count
        and guarded_from_contingency
        == guarded_success_count
    )

    episode_accounting_consistent = (
        planned_episode_count
        == (
            completed_episode_count
            + failed_episode_count
        )
        and planned_episode_count
        == (
            task_count
            * len(conditions)
        )
    )

    pair_accounting_consistent = (
        pair_slots_from_missingness
        == task_count
        and int(
            missingness.get(
                "complete_pairs",
                0,
            )
        )
        == complete_pair_count
    )

    return {
        "schema_version": "1.0",
        "study_id": (
            execution_manifest[
                "study_id"
            ]
        ),
        "task_count": task_count,
        "conditions": conditions,
        "planned_episode_count": (
            planned_episode_count
        ),
        "completed_episode_count": (
            completed_episode_count
        ),
        "failed_episode_count": (
            failed_episode_count
        ),
        "complete_pair_count": (
            complete_pair_count
        ),
        "baseline_success_count": (
            baseline_success_count
        ),
        "guarded_success_count": (
            guarded_success_count
        ),
        "n_11": n_11,
        "n_10": n_10,
        "n_01": n_01,
        "n_00": n_00,
        "baseline_from_contingency": (
            baseline_from_contingency
        ),
        "guarded_from_contingency": (
            guarded_from_contingency
        ),
        "pair_total_from_contingency": (
            contingency_total
        ),
        "marginal_contingency_consistent": (
            marginal_contingency_consistent
        ),
        "pair_accounting_consistent": (
            pair_accounting_consistent
        ),
        "episode_accounting_consistent": (
            episode_accounting_consistent
        ),
        "provider_call_audit": {
            "hosted_model_call_event_count": (
                hosted_model_call_event_count
            ),
            "completed_hosted_model_call_event_count": (
                completed_hosted_model_call_event_count
            ),
            "failed_hosted_model_call_event_count": (
                failed_hosted_model_call_event_count
            ),
            "cache_hit_count": (
                cache_hit_count
            ),
            "cache_miss_count": (
                cache_miss_count
            ),
            "stage_counts": (
                stage_counts
            ),
            "condition_counts": (
                condition_counts
            ),
            "cross_condition_cache_key_reuse_observed": bool(
                cross_condition_cache_keys
            ),
            "cross_condition_cache_keys": (
                cross_condition_cache_keys
            ),
        },
        "all_deterministic_consistency_checks_passed": (
            marginal_contingency_consistent
            and pair_accounting_consistent
            and episode_accounting_consistent
            and not cross_condition_cache_keys
        ),
    }

def analysis_preregistration_fidelity_issues(
    *,
    preregistration: PreregistrationDocument,
    analysis_plan: dict[str, Any],
    analysis_contracts: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Enforce exact machine-readable fidelity between the sealed
    preregistration and the selected analysis executor.
    """
    issues: list[str] = []

    prereg_estimand_id = (
        preregistration.primary_estimand_id
    )

    planned_estimand = analysis_plan.get(
        "estimand"
    )

    if planned_estimand != prereg_estimand_id:
        issues.append(
            "Analysis plan estimand does not exactly match "
            "the sealed preregistered primary_estimand_id."
        )

    matching_contracts = [
        contract
        for contract in analysis_contracts.values()
        if contract.get("estimand")
        == prereg_estimand_id
    ]

    if not matching_contracts:
        issues.append(
            "No registered analysis executor can compute "
            "the sealed preregistered primary_estimand_id."
        )

    selected_executor = analysis_plan.get(
        "analysis_executor"
    )

    selected_contract = analysis_contracts.get(
        str(selected_executor),
        {},
    )

    if (
        selected_contract
        and selected_contract.get("estimand")
        != prereg_estimand_id
    ):
        issues.append(
            "Selected analysis executor does not support "
            "the sealed preregistered primary_estimand_id."
        )

    return sorted(set(issues))

async def create_feasible_experiment_plan(
    *,
    master_prompt: str,
    capability_manifest: dict[str, Any],
    preregistration: PreregistrationDocument,
    repaired_design: RepairedStudyDesign,
    records: list[dict[str, Any]],
    run_dir: Path,
    available_execution_models: list[str],
    maximum_attempts: int = 3,
) -> tuple[
    ExperimentPlan | None,
    dict[str, Any],
]:
    """
    Produce a capability-compliant experiment plan.

    Each generated plan is checked deterministically against the
    frozen capability manifest. Failed plans and reports are retained,
    and the planner receives the exact failures for bounded repair.
    """
    attempts_dir = (
        run_dir
        / "execution"
        / "planning_attempts"
    )
    attempts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_task_count = (
        required_confirmatory_task_count(
            repaired_design
        )
    )

    previous_plan: dict[str, Any] | None = None
    previous_issues: list[str] = []

    final_feasibility: dict[str, Any] = {
        "status": "failed",
        "issue_count": 1,
        "issues": [
            "No experiment-planning attempt completed."
        ],
    }

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        planner_payload: dict[str, Any] = {
            "master_prompt": master_prompt,
            "capability_manifest": (
                capability_manifest
            ),
            "preregistration": (
                preregistration.model_dump()
            ),
            "repaired_design": (
                repaired_design.model_dump()
            ),
            "verified_records": records,
            "available_adapter_families": (
                registered_adapter_families()
            ),
            "available_adapter_contracts": (
                registered_adapter_planning_contracts()
            ),
            "available_execution_models": (
                available_execution_models
            ),
            "required_confirmatory_task_count": (
                required_task_count
            ),
            "planning_attempt": attempt,
            "maximum_planning_attempts": (
                maximum_attempts
            ),
            "hard_execution_constraints": {
                "human_scientific_labour_allowed": (
                    capability_manifest.get(
                        "human_scientific_labour_allowed",
                        False,
                    )
                ),
                "external_partner_allowed": (
                    capability_manifest.get(
                        "external_partner_allowed",
                        False,
                    )
                ),
                "human_annotation_allowed": (
                    capability_manifest.get(
                        "human_annotation_allowed",
                        False,
                    )
                ),
                "manual_adjudication_allowed": (
                    capability_manifest.get(
                        "manual_adjudication_allowed",
                        False,
                    )
                ),
                "nda_resources_allowed": (
                    capability_manifest.get(
                        "nda_resources_allowed",
                        False,
                    )
                ),
                "private_live_lab_available": (
                    capability_manifest.get(
                        "private_live_lab_available",
                        False,
                    )
                ),
                "public_datasets_only": (
                    capability_manifest.get(
                        "public_datasets_only",
                        False,
                    )
                ),
                "autonomous_scoring_required": (
                    capability_manifest.get(
                        "autonomous_scoring_required",
                        False,
                    )
                ),
                "docker_available": (
                    capability_manifest.get(
                        "docker_available",
                        False,
                    )
                ),
                "kubernetes_available": (
                    capability_manifest.get(
                        "kubernetes_available",
                        False,
                    )
                ),
                "hosted_model_api_available": (
                    capability_manifest.get(
                        "hosted_model_api_available",
                        False,
                    )
                ),
                "cpu_execution_available": (
                    capability_manifest.get(
                        "cpu_execution_available",
                        False,
                    )
                ),
                "local_gpu": (
                    capability_manifest.get(
                        "local_gpu",
                        {
                            "available": False,
                            "memory_gb": 0,
                        },
                    )
                ),
                "maximum_planned_model_calls": (
                    capability_manifest.get(
                        "maximum_planned_model_calls"
                    )
                ),
                "maximum_wall_clock_days": (
                    capability_manifest.get(
                        "maximum_wall_clock_days"
                    )
                ),
            },
            "mandatory_planning_instruction": (
                "Treat the frozen capability manifest as a hard "
                "execution contract. Every model, execution batch, "
                "validator, scorer, transformation, fallback and "
                "dependency must be executable with the listed "
                "capabilities. Do not include unavailable resources "
                "as optional, recommended, audit, validation or "
                "future components. When no local GPU is available, "
                "use hosted model APIs or CPU-compatible methods. "
                "When human labour is prohibited, all labels, audits, "
                "scoring and validation must be autonomous. Set "
                "adapter_family to exactly one identifier from "
                "available_adapter_families. Do not invent, describe, "
                "expand, rename, or decorate the identifier. The "
                "implementation strategy, resources, batches, and "
                "result schema must fit the selected registered "
                "adapter's actual scope. "
                "The input also contains available_adapter_contracts. "
                "When selecting an adapter_family, populate every "
                "machine-readable execution-contract field in "
                "ExperimentPlan so that it satisfies that adapter's "
                "contract exactly. Do not encode required adapter "
                "contract values only in prose fields. "
                "The input also contains available_execution_models. "
                "Set model_name to exactly one identifier from that "
                "list. Do not invent, rename, qualify, or substitute "
                "a hosted model identifier. Set model_version equal "
                "to the selected model_name unless an explicitly "
                "different version is supplied by the capability "
                "contract. "
                "The input contains required_confirmatory_task_count, "
                "which is the machine-readable sample size selected by "
                "the repaired scientific design. Set task_count exactly "
                "to required_confirmatory_task_count. Do not reduce the "
                "confirmatory sample size merely to fit an adapter. If "
                "the required sample cannot fit the frozen capabilities, "
                "the plan is infeasible rather than a smaller study."
            ),
        }

        if previous_plan is not None:
            planner_payload[
                "rejected_previous_plan"
            ] = previous_plan

            planner_payload[
                "deterministic_feasibility_issues"
            ] = previous_issues

            planner_payload[
                "repair_instruction"
            ] = (
                "Repair every deterministic feasibility failure. "
                "Remove or replace the offending dependency; do not "
                "rename it or retain it as optional. Remove all local "
                "GPU, CUDA, LoRA, local 7B or 70B model, human-rater, "
                "expert-review, annotation, manual-adjudication, "
                "external-partner, NDA, private-lab and unavailable "
                "Kubernetes requirements. Preserve the scientific "
                "question and estimands where executable. Set "
                "adapter_family to exactly one identifier from "
                "available_adapter_families; never invent or decorate "
                "an adapter identifier. Preserve "
                "required_confirmatory_task_count exactly; do not repair "
                "a capability conflict by silently reducing the sealed "
                "scientific sample size."
            )

        experiment_plan = await run_agent(
            EXPERIMENT_PLANNER,
            planner_payload,
            expected_type=ExperimentPlan,
            stage_name=(
                "Experiment planning "
                f"attempt {attempt}"
            ),
        )

        # Canonicalize adapter-owned mechanical execution fields.
        # These values are fixed by the registered adapter contract, not
        # scientific choices delegated to the autonomous planner.
        selected_adapter_contract = (
            registered_adapter_planning_contracts().get(
                experiment_plan.adapter_family
            )
        )

        if selected_adapter_contract is not None:
            experiment_plan.initial_generation_calls_per_task = int(
                selected_adapter_contract[
                    "initial_generation_calls_per_task"
                ]
            )
            experiment_plan.maximum_repair_calls_per_task = int(
                selected_adapter_contract[
                    "maximum_repair_calls_per_task"
                ]
            )
            experiment_plan.maximum_model_calls_per_task = int(
                selected_adapter_contract[
                    "maximum_model_calls_per_task"
                ]
            )

        plan_dict = (
            experiment_plan.model_dump()
        )

        write_json(
            attempts_dir
            / (
                "experiment_plan_attempt_"
                f"{attempt:02d}.json"
            ),
            plan_dict,
        )

        generic_feasibility = feasibility_report(
            design=plan_dict,
            capability_manifest=capability_manifest,
        )

        combined_issues = list(
            generic_feasibility.get(
                "issues",
                [],
            )
        )

        combined_issues.extend(
            adapter_compatibility_issues(
                plan_dict
            )
        )

        if plan_dict.get("study_id") != preregistration.study_id:
            combined_issues.append(
                "ExperimentPlan.study_id must equal the "
                "preregistration study_id exactly."
            )

        if plan_dict.get("task_count") != required_task_count:
            combined_issues.append(
                "task_count must equal the machine-readable "
                "required_confirmatory_task_count "
                f"({required_task_count})."
            )

        if plan_dict.get("model_name") not in available_execution_models:
            combined_issues.append(
                "model_name must be exactly one identifier from "
                "available_execution_models."
            )

        if (
            plan_dict.get("model_name")
            and plan_dict.get("model_version")
            != plan_dict.get("model_name")
        ):
            combined_issues.append(
                "model_version must equal model_name for the "
                "currently available hosted execution model."
            )

        combined_issues = sorted(
            set(combined_issues)
        )

        final_feasibility = {
            "status": (
                "passed"
                if not combined_issues
                else "failed"
            ),
            "issue_count": len(combined_issues),
            "issues": combined_issues,
        }

        write_json(
            attempts_dir
            / (
                "experiment_plan_attempt_"
                f"{attempt:02d}_feasibility.json"
            ),
            final_feasibility,
        )

        if (
            final_feasibility["status"]
            == "passed"
        ):
            return (
                experiment_plan,
                final_feasibility,
            )

        previous_plan = plan_dict
        previous_issues = list(
            final_feasibility.get(
                "issues",
                [],
            )
        )

    return (
        None,
        final_feasibility,
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

        register_builtin_execution_adapters()
        register_builtin_analysis_executors()

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
        paper_run_constraints: dict[str, Any],
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
                "paper_run_constraints": (
                    paper_run_constraints
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
            "manuscript/revision_rounds",
            "manuscript/final",
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

        available_analysis_families = (
            registered_analysis_families()
        )

        available_analysis_contracts = (
            registered_analysis_planning_contracts()
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
            "available_analysis_families": (
                available_analysis_families
            ),
            "available_analysis_contracts": (
                available_analysis_contracts
            ),
            "available_adapter_families": (
                registered_adapter_families()
            ),
            "available_adapter_contracts": (
                registered_adapter_planning_contracts()
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

        evidence_alias_index = (
            build_evidence_alias_index(
                records
            )
        )

        unknown_repair_evidence_ids = sorted(
            evidence_id
            for evidence_id
            in repaired_design.evidence_record_ids
            if (
                normalise_evidence_id(
                    evidence_id
                )
                not in evidence_alias_index
            )
        )

        if unknown_repair_evidence_ids:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                ],
                failed_gate=(
                    "Autonomous design repair referenced "
                    "evidence that was not retrieved."
                ),
                final_state=(
                    "AUTONOMOUS_EVIDENCE_REPAIR_REQUIRED"
                ),
                warnings=[
                    (
                        "Unresolved repaired-design evidence IDs: "
                        + ", ".join(
                            unknown_repair_evidence_ids
                        )
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
                selected_candidate_id=(
                    selected_candidate_id
                ),
                development_rehearsal=(
                    self.development_rehearsal
                ),
                additional_fields={
                    "unknown_repair_evidence_ids": (
                        unknown_repair_evidence_ids
                    ),
                },
            )

            return report

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

        available_adapter_contracts = (
            registered_adapter_planning_contracts()
        )

        repaired_design_dict = (
            repaired_design.model_dump()
        )

        repaired_design_feasibility = (
            feasibility_report(
                design=(
                    repaired_design_dict
                ),
                capability_manifest=(
                    capability_manifest
                ),
            )
        )

        adapter_scientific_issues = (
            repaired_design_adapter_capability_issues(
                repaired_design_dict,
                available_adapter_contracts=(
                    available_adapter_contracts
                ),
            )
        )

        if adapter_scientific_issues:
            combined_issues = sorted(
                set(
                    list(
                        repaired_design_feasibility.get(
                            "issues",
                            [],
                        )
                    )
                    + adapter_scientific_issues
                )
            )

            repaired_design_feasibility = {
                "status": "failed",
                "issue_count": len(combined_issues),
                "issues": combined_issues,
            }

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
        # 5. Generate provisional preregistration
        # -------------------------------------------------

        available_adapter_contracts = (
            registered_adapter_planning_contracts()
        )

        available_execution_models = [
            self.model
        ]

        prereg_required_task_count = (
            required_confirmatory_task_count(
                repaired_design
            )
        )

        preregistration = None
        preregistration_contract_issues: list[str] = []

        maximum_preregistration_attempts = 3



        for prereg_attempt in range(
            1,
            maximum_preregistration_attempts + 1,
        ):
            preregistration = await run_agent(
                PREREGISTRATION_AGENT,
                {
                    "master_prompt": master_prompt,
                    "capability_manifest": (
                        capability_manifest
                    ),
                    "selected_candidate_id": (
                        selected_candidate_id
                    ),
                    "repaired_design": (
                        repaired_design.model_dump()
                    ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "verified_record_ids": (
                        allowed_evidence_record_ids
                    ),
                    "registered_adapter_planning_contracts": (
                        available_adapter_contracts
                    ),
                    "available_analysis_families": (
                        available_analysis_families
                    ),
                    "available_analysis_contracts": (
                        available_analysis_contracts
                    ),
                    "available_execution_models": (
                        available_execution_models
                    ),
                    "required_confirmatory_task_count": (
                        prereg_required_task_count
                    ),
                    "previous_contract_issues": (
                        preregistration_contract_issues
                    ),
                    "instruction": (
                        "Produce a complete provisional "
                        "preregistration that is exactly "
                        "executable under one supplied "
                        "registered adapter planning contract. "
                        "The preregistered confirmatory primary "
                        "estimand must also be exactly executable "
                        "by one supplied registered analysis "
                        "planning contract. Set primary_estimand_id "
                        "to the exact machine-readable estimand "
                        "identifier from that analysis contract. "
                        "Keep primary_estimand as the scientifically "
                        "readable description of the same estimand. "
                        "Do not invent, paraphrase, or substitute "
                        "primary_estimand_id. "
                        "Use only the supplied available "
                        "execution models. The structured "
                        "execution_contract, model_scope, "
                        "transformation_scope, sampling plan, "
                        "and analysis plan must describe the "
                        "same experiment. Do not introduce additional model "
                        "families, conditions, transformations, run counts, "
                        "execution modes, consensus stages, ensembles, multi-model "
                        "voting, simulated-human gates, prompt-family strata, or "
                        "any other defense stage absent from the selected executable "
                        "adapter contract. Scientific prose, hypotheses, estimands, "
                        "sampling plans, and analysis plans must describe only stages "
                        "and task strata that the selected adapter actually executes "
                        "and records. Treat every field of the supplied executable "
                        "adapter contract as authoritative. In particular, when "
                        "retrieval_augmented_generation is false, do not describe "
                        "retrieval-augmented generation, RAG, retrieval augmentation, "
                        "or a retrieval stage as part of the registered experiment, "
                        "intervention, hypothesis, estimand, sampling plan, or analysis "
                        "plan. Such methods may exist in the literature, but they are "
                        "not executed by this experiment unless the adapter contract "
                        "explicitly enables them. Set task_count exactly to "
                        "required_confirmatory_task_count. "
                        "If previous_contract_issues is nonempty, "
                        "return a complete replacement "
                        "preregistration correcting all of them."
                    ),
                },
                expected_type=(
                    PreregistrationDocument
                ),
                stage_name=(
                    "Provisional preregistration "
                    f"attempt {prereg_attempt}"
                ),
            )

            preregistration = (
                canonicalize_preregistration_execution_contract(
                    preregistration,
                    planning_contracts=(
                        available_adapter_contracts
                    ),
                    available_execution_models=(
                        available_execution_models
                    ),
                    required_task_count=(
                        prereg_required_task_count
                    ),
                )
            )

            preregistration = (
                canonicalize_preregistration_analysis_contract(
                    preregistration,
                    analysis_contracts=(
                        available_analysis_contracts
                    ),
                )
            )

            # Persist the exact fully canonical preregistration object
            # that is about to be evaluated by the blocking execution-contract
            # gate. This makes every bounded preregistration rejection
            # reproducible from archived run artifacts.
            write_json(
                run_dir
                / "preregistration"
                / (
                    "preregistration_canonical_attempt_"
                    f"{prereg_attempt:02d}.json"
                ),
                preregistration.model_dump(mode="json"),
            )

            preregistration_contract_issues = (
                preregistration_execution_contract_issues(
                    preregistration,
                    planning_contracts=(
                        available_adapter_contracts
                    ),
                    available_execution_models=(
                        available_execution_models
                    ),
                    required_task_count=(
                        prereg_required_task_count
                    ),
                )
            )

            preregistration_contract_issues.extend(
                preregistration_analysis_contract_issues(
                    preregistration,
                    analysis_contracts=(
                        available_analysis_contracts
                    ),
                )
            )

            preregistration_contract_issues.extend(
                preregistration_scientific_coherence_issues(
                    preregistration
                )
            )

            preregistration_contract_issues.extend(
                preregistration_identifiability_issues(
                    preregistration
                )
            )

            preregistration_contract_issues = sorted(
                set(preregistration_contract_issues)
            )

            write_json(
                run_dir
                / "preregistration"
                / (
                    "preregistration_contract_check_"
                    f"{prereg_attempt:02d}.json"
                ),
                {
                    "attempt": prereg_attempt,
                    "passed": not (
                        preregistration_contract_issues
                    ),
                    "issues": (
                        preregistration_contract_issues
                    ),
                    "declared_execution_contract": (
                        preregistration
                        .execution_contract
                        .model_dump()
                    ),
                    "declared_primary_estimand_id": (
                        preregistration.primary_estimand_id
                    ),
                    "declared_primary_estimand": (
                        preregistration.primary_estimand
                    ),
                    "available_analysis_estimands": sorted(
                        {
                            contract.get("estimand")
                            for contract
                            in available_analysis_contracts.values()
                            if isinstance(
                                contract.get("estimand"),
                                str,
                            )
                        }
                    ),
                },
            )

            if not preregistration_contract_issues:
                break

        if (
            preregistration is None
            or preregistration_contract_issues
        ):
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                ],
                failed_gate=(
                    "Provisional preregistration could "
                    "not satisfy the executable adapter "
                    "contract after bounded autonomous repair."
                ),
                final_state=(
                    "PREREGISTRATION_EXECUTION_CONTRACT_FAILED"
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
                    "preregistration_contract_issues": (
                        preregistration_contract_issues
                    ),
                },
            )

            return report

        preregistration_path = (
            run_dir
            / "preregistration"
            / "preregistration.json"
        )

        write_json(
            preregistration_path,
            preregistration,
        )

        preregistration_hash_path = (
            preregistration_path.parent
            / "preregistration.sha256"
        )

        # A preregistration is not sealed until the
        # experiment plan passes deterministic feasibility.
        if preregistration_hash_path.exists():
            preregistration_hash_path.unlink()

        # -------------------------------------------------
        # 6. Experiment planning with bounded repair
        # -------------------------------------------------

        available_adapter_families = (
            registered_adapter_families()
        )

        if not available_adapter_families:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                ],
                failed_gate=(
                    "No autonomous execution adapters "
                    "are registered."
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
                additional_fields={
                    "available_adapter_families": [],
                },
            )

            return report

        (
            experiment_plan,
            experiment_feasibility,
        ) = await create_feasible_experiment_plan(
            master_prompt=master_prompt,
            capability_manifest=(
                capability_manifest
            ),
            preregistration=preregistration,
            repaired_design=repaired_design,
            records=records,
            run_dir=run_dir,
            available_execution_models=(
                available_execution_models
            ),
            maximum_attempts=3,
        )

        write_json(
            design_dir
            / "experiment_plan_feasibility.json",
            experiment_feasibility,
        )

        if experiment_plan is None:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                ],
                failed_gate=(
                    "The autonomous experiment planner "
                    "could not produce a capability-compliant "
                    "plan after bounded repair attempts."
                ),
                final_state=(
                    "AUTONOMOUS_EXPERIMENT_PLAN_REPAIR_REQUIRED"
                ),
                warnings=list(
                    experiment_feasibility.get(
                        "issues",
                        [],
                    )
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
                    "experiment_plan_repair_attempts": 3,
                    "feasibility_issues": list(
                        experiment_feasibility.get(
                            "issues",
                            [],
                        )
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 7. Accept plan and seal preregistration
        # -------------------------------------------------

        experiment_plan_path = (
            run_dir
            / "execution"
            / "experiment_plan.json"
        )

        write_json(
            experiment_plan_path,
            experiment_plan,
        )

        preregistration_hash = (
            sha256_file(
                preregistration_path
            )
        )

        preregistration_hash_path.write_text(
            preregistration_hash + "\n",
            encoding="utf-8",
        )

        write_json(
            run_dir
            / "preregistration"
            / "sealing_manifest.json",
            {
                "preregistration_sha256": (
                    preregistration_hash
                ),
                "selected_candidate_id": (
                    selected_candidate_id
                ),
                "repaired_design_feasibility": (
                    "passed"
                ),
                "experiment_plan_feasibility": (
                    "passed"
                ),
                "experiment_plan_path": str(
                    experiment_plan_path.relative_to(
                        run_dir
                    )
                ),
                "sealed_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
        )

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
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
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

        execution_manifest_issues = (
            validate_execution_manifest(
                execution_manifest,
                plan=(
                    experiment_plan.model_dump()
                ),
                output_dir=(
                    run_dir / "execution"
                ),
                maximum_model_calls=(
                    capability_manifest.get(
                        "maximum_planned_model_calls"
                    )
                ),
            )
        )

        if execution_manifest_issues:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
                    "execution_adapter_resolved",
                ],
                failed_gate=(
                    "Autonomous execution adapter "
                    "did not produce a valid completed "
                    "execution manifest."
                ),
                final_state=(
                    "AUTONOMOUS_EXECUTION_INCOMPLETE"
                ),
                warnings=(
                    execution_manifest_issues
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
                    "execution_manifest_issues": (
                        execution_manifest_issues
                    ),
                },
            )

            return report

        # -------------------------------------------------
        # 10. Analysis planning and execution
        # -------------------------------------------------

        if not available_analysis_families:
            report = create_failure_report(
                passed_gates=[
                    "fresh_run",
                    "autonomous_discovery",
                    "candidate_validation",
                    "evidence_verification",
                    "autonomous_design_repair",
                    "repaired_design_feasibility",
                    "provisional_preregistration",
                    "experiment_plan",
                    "experiment_plan_feasibility",
                    "sealed_preregistration",
                    "execution_adapter_resolved",
                    "execution_completed",
                ],
                failed_gate=(
                    "No deterministic analysis executors "
                    "are registered."
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
                additional_fields={
                    "available_analysis_families": [],
                },
            )

            return report

        analysis_attempts_dir = (
            run_dir
            / "analysis"
            / "planning_attempts"
        )
        analysis_attempts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        previous_analysis_plan: dict[str, Any] | None = None
        previous_analysis_issues: list[str] = []
        analysis_plan: AnalysisPlan | None = None

        for attempt in range(1, 4):
            analysis_payload: dict[str, Any] = {
                "master_prompt": master_prompt,
                "capability_manifest": capability_manifest,
                "preregistration": preregistration.model_dump(),
                "experiment_plan": experiment_plan.model_dump(),
                "execution_manifest": (
                    _compact_execution_manifest_for_analysis_planning(
                        execution_manifest
                    )
                ),
                "available_analysis_families": (
                    available_analysis_families
                ),
                "available_analysis_contracts": (
                    available_analysis_contracts
                ),
            }

            if previous_analysis_plan is not None:
                analysis_payload[
                    "rejected_previous_analysis_plan"
                ] = previous_analysis_plan
                analysis_payload[
                    "deterministic_analysis_issues"
                ] = previous_analysis_issues
                analysis_payload[
                    "repair_instruction"
                ] = (
                    "Repair every deterministic analysis compatibility "
                    "failure while preserving the sealed preregistration. "
                    "Do not substitute a different primary estimand merely "
                    "to match an available executor. If no registered "
                    "analysis executor can compute the sealed preregistered "
                    "primary estimand, preserve that incompatibility rather "
                    "than changing the estimand. "
                    "Use exact machine-readable identifiers from the "
                    "selected available_analysis_contracts entry. Do not "
                    "paraphrase estimand or failed-call-treatment "
                    "identifiers."
                )

            candidate_analysis_plan = await run_agent(
                ANALYSIS_PLANNER,
                analysis_payload,
                expected_type=AnalysisPlan,
                stage_name=(
                    "Analysis planning "
                    f"attempt {attempt}"
                ),
            )

            candidate_dict = (
                candidate_analysis_plan.model_dump()
            )

            write_json(
                analysis_attempts_dir
                / (
                    "analysis_plan_attempt_"
                    f"{attempt:02d}.json"
                ),
                candidate_dict,
            )

            issues = analysis_compatibility_issues(
                analysis_plan=candidate_dict,
                execution_manifest=execution_manifest,
            )

            issues.extend(
                analysis_preregistration_fidelity_issues(
                    preregistration=preregistration,
                    analysis_plan=candidate_dict,
                    analysis_contracts=(
                        available_analysis_contracts
                    ),
                )
            )

            issues = sorted(set(issues))

            write_json(
                analysis_attempts_dir
                / (
                    "analysis_plan_attempt_"
                    f"{attempt:02d}_compatibility.json"
                ),
                {
                    "compatible": not issues,
                    "issues": issues,
                },
            )

            if not issues:
                analysis_plan = candidate_analysis_plan
                break

            previous_analysis_plan = candidate_dict
            previous_analysis_issues = list(issues)

        if analysis_plan is None:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                ],
                failed_gate=(
                    "The autonomous analysis planner could not "
                    "produce an executor-compatible analysis plan "
                    "after bounded repair attempts."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_PLAN_REPAIR_REQUIRED"
                ),
                warnings=previous_analysis_issues,
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
                    "analysis_compatibility_issues": (
                        previous_analysis_issues
                    ),
                },
            )

            return report

        write_json(
            run_dir
            / "analysis"
            / "analysis_plan.json",
            analysis_plan,
        )

        analysis_executor = (
            resolve_analysis_executor(
                analysis_plan=(
                    analysis_plan.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
            )
        )

        if analysis_executor is None:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                    "analysis_plan",
                ],
                failed_gate=(
                    "No installed deterministic "
                    "analysis executor supports the "
                    "analysis plan and execution output."
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

        analysis_results = (
            analysis_executor.execute(
                analysis_plan=(
                    analysis_plan.model_dump()
                ),
                preregistration=(
                    preregistration.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
                run_dir=run_dir,
            )
        )

        results_path = (
            run_dir
            / "analysis"
            / "results.json"
        )

        write_json(
            results_path,
            analysis_results,
        )

        analysis_result_issues = (
            validate_analysis_results(
                analysis_results,
                run_dir=run_dir,
                execution_manifest=(
                    execution_manifest
                ),
            )
        )

        if analysis_result_issues:
            report = create_failure_report(
                passed_gates=[
                    "execution_completed",
                    "analysis_plan",
                    "analysis_executor_resolved",
                ],
                failed_gate=(
                    "Deterministic analysis executor "
                    "did not produce valid completed "
                    "analysis results."
                ),
                final_state=(
                    "AUTONOMOUS_ANALYSIS_INCOMPLETE"
                ),
                warnings=(
                    analysis_result_issues
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
                    "analysis_result_issues": (
                        analysis_result_issues
                    ),
                },
            )

            return report

        deterministic_reconciliation = (
            build_deterministic_reconciliation(
                experiment_plan=(
                    experiment_plan.model_dump()
                ),
                execution_manifest=(
                    execution_manifest
                ),
                analysis_results=(
                    analysis_results
                ),
                run_dir=run_dir,
            )
        )

        write_json(
            run_dir
            / "analysis"
            / "deterministic_reconciliation.json",
            deterministic_reconciliation,
        )

        manuscript_evidence_bundle = (
            build_manuscript_evidence_bundle(
                run_dir=run_dir,
                execution_manifest=(
                    execution_manifest
                ),
                analysis_results=(
                    analysis_results
                ),
            )
        )

        write_json(
            run_dir
            / "manuscript"
            / "manuscript_evidence_bundle.json",
            manuscript_evidence_bundle,
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
                "verified_records": (
                    compact_verified_records_for_manuscript(
                        records
                    )
                ),
                "evidence_synthesis": (
                    _compact_evidence_synthesis_for_manuscript(
                        synthesis
                    )
                ),
                "evidence_verification": (
                    evidence_report
                ),
                "preregistration": (
                    preregistration.model_dump()
                ),
                "execution_manifest": (
                    compact_execution_manifest_for_manuscript(
                        execution_manifest
                    )
                ),
                "analysis_plan": (
                    analysis_plan.model_dump()
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
                "manuscript_evidence_bundle": (
                    manuscript_evidence_bundle
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
        # 12-13. Bounded autonomous peer review / revision
        # -------------------------------------------------

        review_rounds_dir = (
            run_dir
            / "manuscript"
            / "review_rounds"
        )
        review_rounds_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        revision_rounds_dir = (
            run_dir
            / "manuscript"
            / "revision_rounds"
        )
        revision_rounds_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_manuscript = draft
        manuscript_revision_context = (
            _manuscript_revision_context(
                records=records,
                execution_manifest=execution_manifest,
                manuscript_evidence_bundle=(
                    manuscript_evidence_bundle
                ),
                evidence_synthesis=synthesis,
            )
        )

        latest_peer_review: PeerReviewReport | None = None

        maximum_peer_review_rounds = 5

        for review_round in range(
            1,
            maximum_peer_review_rounds + 1,
        ):
            latest_peer_review = await run_agent(
                PEER_REVIEWER,
                {
                    "master_prompt": master_prompt,
                    "evidence_synthesis": (
                        _compact_evidence_synthesis_for_manuscript(
                            synthesis
                        )
                    ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                        _compact_execution_manifest_for_manuscript(
                            execution_manifest
                        )
                    ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_evidence_bundle
                    ),
                    "manuscript": (
                        current_manuscript.model_dump()
                    ),
                    "review_round": review_round,
                },
                expected_type=PeerReviewReport,
                stage_name=(
                    "AI peer review "
                    f"round {review_round}"
                ),
            )

            write_json(
                review_rounds_dir
                / f"review_{review_round:02d}.json",
                latest_peer_review,
            )

            review_is_finalisable = (
                latest_peer_review.accept_for_finalisation
                and not latest_peer_review.critical_issues
                and not latest_peer_review.required_revisions
            )

            if review_is_finalisable:
                break

            if (
                review_round
                >= maximum_peer_review_rounds
            ):
                break

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "master_prompt": master_prompt,
                    "verified_records": (
                            manuscript_revision_context[
                                "verified_records"
                            ]
                        ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                            manuscript_revision_context[
                                "execution_manifest"
                            ]
                        ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_revision_context[
                            "manuscript_evidence_bundle"
                        ]
                    ),
                    "evidence_synthesis": (
                        manuscript_revision_context[
                            "evidence_synthesis"
                        ]
                    ),
                    "manuscript": (
                        current_manuscript
                        .model_dump()
                    ),
                    "peer_review": (
                        latest_peer_review
                        .model_dump()
                    ),
                    "revision_round": (
                        review_round
                    ),
                    "revision_instruction": (
                        """
                        Address every required revision that can be
                        resolved from existing verified evidence,
                        execution artifacts, and analysis results.
                        Do not invent new experiments, data, references,
                        statistics, URLs, artifact locations, or empirical
                        claims. If a reviewer request cannot be resolved
                        from existing artifacts, preserve it explicitly
                        as an unresolved limitation rather than fabricating
                        support.

                        Treat deterministic_reconciliation
                        as authoritative for arithmetic
                        consistency, execution accounting,
                        and observed cache-key reuse. Do not
                        adopt a reviewer claim that contradicts
                        a passing deterministic reconciliation
                        check. If reviewer prose conflicts with
                        the deterministic artifact, preserve the
                        artifact-backed facts and explicitly
                        resolve the reviewer concern from those
                        facts.

                        A compact manuscript_evidence_bundle may be supplied. Treat it as
                        authoritative for artifact-grounded manuscript details such as
                        representative scoring examples, canonical artifact paths and hashes,
                        contamination-summary outputs, paired contingency results, execution
                        accounting, and reproducibility details. When a reviewer requests such
                        information and it is present in this bundle, incorporate the actual
                        value into the manuscript rather than merely stating that an artifact
                        exists.
                        """
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Autonomous manuscript revision "
                    f"round {review_round}"
                ),
            )

            write_json(
                revision_rounds_dir
                / (
                    "revised_package_"
                    f"{review_round:02d}.json"
                ),
                revised_manuscript,
            )

            current_manuscript = (
                revised_manuscript
            )

        if latest_peer_review is None:
            raise RuntimeError(
                "Autonomous peer review produced "
                "no review artifact."
            )

        revised_manuscript = (
            current_manuscript
        )

        write_json(
            run_dir
            / "manuscript"
            / "revised_package.json",
            revised_manuscript,
        )

        # -------------------------------------------------
        # 14. Deterministic IEEE rendering and page validation
        # -------------------------------------------------

        publication_dir = (
            run_dir
            / "manuscript"
            / "final"
        )

        maximum_format_revision_rounds = 16
        publication_validation: dict[str, Any] | None = None

        best_manuscript = revised_manuscript
        best_publication_validation: dict[str, Any] | None = None
        best_page_count = -1
        best_section_word_count = manuscript_section_word_count(
            revised_manuscript
        )

        for format_round in range(
            0,
            maximum_format_revision_rounds + 1,
        ):
            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                publication_dir
                / (
                    "publication_validation_"
                    f"{format_round:02d}.json"
                ),
                publication_validation,
            )

            current_page_count = publication_validation.get(
                "page_count"
            )
            current_maximum_pages = publication_validation.get(
                "maximum_pages"
            )

            if (
                isinstance(current_page_count, int)
                and isinstance(current_maximum_pages, int)
                and current_page_count <= current_maximum_pages
            ):
                if current_page_count > best_page_count:
                    best_page_count = current_page_count
                    best_manuscript = revised_manuscript
                    best_publication_validation = dict(
                        publication_validation
                    )
                    best_section_word_count = (
                        manuscript_section_word_count(
                            revised_manuscript
                        )
                    )

                elif (
                    current_page_count == best_page_count
                    and manuscript_section_word_count(
                        revised_manuscript
                    )
                    > best_section_word_count
                ):
                    best_manuscript = revised_manuscript
                    best_publication_validation = dict(
                        publication_validation
                    )
                    best_section_word_count = (
                        manuscript_section_word_count(
                            revised_manuscript
                        )
                    )

            if publication_validation.get(
                "passed"
            ):
                break

            compile_status = (
                publication_validation.get(
                    "compile_status"
                )
            )

            # Manuscript revision is appropriate only for a
            # successfully compiled paper whose compiled page count
            # does not exactly match the frozen IEEE page budget.
            # Compilation failures are infrastructure/rendering
            # failures, not reasons to alter scientific manuscript
            # content.
            if compile_status != "passed":
                break

            page_count = (
                publication_validation.get(
                    "page_count"
                )
            )
            maximum_pages = (
                publication_validation.get(
                    "maximum_pages"
                )
            )

            uses_full_page_budget = (
                publication_validation.get(
                    "uses_full_page_budget",
                    False,
                )
            )

            if uses_full_page_budget:
                break

            if (
                page_count is None
                or maximum_pages is None
            ):
                break

            if (
                format_round
                >= maximum_format_revision_rounds
            ):
                break

            if page_count > maximum_pages:
                revision_instruction = (
                    f"The manuscript compiled successfully but is "
                    f"{page_count} pages, exceeding the frozen IEEE "
                    f"conference budget of {maximum_pages} pages. "
                    f"Revise it to occupy exactly {maximum_pages} "
                    "pages. Shorten and compact the manuscript while "
                    "preserving supported scientific claims, the "
                    "primary methods and results, reviewer-resolved "
                    "information, required references, reproducibility "
                    "information, and the mandatory Disclosure "
                    "Statement. Prefer concise scientific phrasing and "
                    "efficient presentation over deleting substantive "
                    "technical material. Do not change empirical "
                    "results, add unsupported claims, remove required "
                    "disclosure content, manipulate the IEEE template, "
                    "shrink fonts or margins, or invent new evidence."
                )
            else:
                pages_missing = maximum_pages - page_count

                # Overshoot-first convergence: while the paper is
                # underfilled, deliberately target one page above the frozen
                # budget. Once an over-limit candidate exists, the next round
                # compacts that actual candidate toward exactly five pages.
                overshoot_target_pages = maximum_pages + 1
                current_section_words = manuscript_section_word_count(
                    best_manuscript
                )
                minimum_expansion_words = max(
                    2300,
                    int(current_section_words * 1.45),
                )
                preferred_expansion_words = max(
                    2600,
                    int(current_section_words * 1.65),
                )

                substantive_expansion_target = (
                    f"Do NOT target {maximum_pages} pages from below in this "
                    f"underfill phase. Deliberately target approximately "
                    f"{overshoot_target_pages} compiled IEEE pages so that a "
                    "later round can compact from above. Produce at least "
                    f"{minimum_expansion_words} substantive words across the "
                    "scientific sections, preferably around "
                    f"{preferred_expansion_words} words when the archived "
                    "evidence supports that level of detail. This word floor "
                    "is a page-convergence control, not permission to pad: "
                    "every added sentence must communicate distinct "
                    "artifact-grounded scientific information. Add at least "
                    "eight distinct new artifact-grounded paragraphs across "
                    "Methods, Results, Discussion/Limitations, reproducibility, "
                    "and related work, plus one or more compact evidence-grounded "
                    "tables where supported. Preserve all supported content "
                    "already present. A revision that remains shorter than the "
                    "best underfilled manuscript is not useful for convergence. "
                )

                revision_instruction = (
                    f"The manuscript compiled successfully but occupies only "
                    f"{page_count} of the required {maximum_pages} IEEE pages, "
                    f"leaving {pages_missing} full page(s) of the scientific "
                    f"page budget unused. The final paper must occupy exactly "
                    f"{maximum_pages} compiled pages, including references and "
                    f"the mandatory Disclosure Statement. This is format "
                    f"revision round {format_round + 1} of "
                    f"{maximum_format_revision_rounds}. "
                    "Substantively expand the manuscript using only information "
                    "supported by the archived autonomous-run artifacts and "
                    "verified evidence. "
                    "Make structural additions rather than primarily rewriting "
                    "existing prose. Add distinct artifact-grounded material across "
                    "multiple scientific sections. Where supported, add methodological "
                    "detail, execution accounting, quantitative interpretation, "
                    "reproducibility information, limitations, and compact "
                    "artifact-grounded tables or figures. Preserve existing supported "
                    "paragraphs rather than exchanging them for new wording. "
                    f"{substantive_expansion_target}"
                    "Preserve all already artifact-supported reviewer-resolved "
                    "content from the current manuscript. While the manuscript "
                    "remains below the required page count, do not shorten, "
                    "remove, or replace substantive supported material merely "
                    "to improve concision. Each underfill revision must be "
                    "cumulative: retain existing Methods, Results, tables, "
                    "evidence mappings, reproducibility details, limitations, "
                    "Disclosure content, and resolved reviewer clarifications, "
                    "then add further missing artifact-grounded scientific "
                    "material."
                    "Do not merely rephrase existing text or "
                    "make small stylistic edits; add materially useful scientific "
                    "content that is currently absent, compressed, or insufficiently "
                    "explained. Prioritize, where supported by the available "
                    "artifacts: detailed methodology and execution semantics; "
                    "experimental design and preregistration rationale; complete "
                    "quantitative results; statistical interpretation; execution "
                    "and failure accounting; representative artifact-grounded "
                    "examples or diagnostics; reviewer-requested clarifications; "
                    "deviations and missingness; limitations and threats to "
                    "validity; operational implications; reproducibility details; "
                    "and additional verified related-work context. Use tables or "
                    "figures when they communicate existing artifact-grounded "
                    "results more effectively than prose. Do not invent "
                    "experiments, observations, statistics, citations, examples, "
                    "repositories, artifact locations, or claims. Do not pad with "
                    "verbosity, repetition, formatting tricks, artificial spacing, "
                    "or unsupported filler. The objective is a genuinely "
                    "complete, dense scientific paper. In this underfill phase, "
                    f"aim to cross the boundary at approximately {overshoot_target_pages} "
                    "compiled pages; exact five-page convergence will then occur by "
                    "compaction from above."
                )

            format_feedback = {
                "page_count": page_count,
                "maximum_pages": maximum_pages,
                "current_section_word_count": (
                    manuscript_section_word_count(
                        best_manuscript
                    )
                ),
                "underfill_overshoot_target_pages": (
                    maximum_pages + 1
                    if page_count < maximum_pages
                    else None
                ),
                "within_page_limit": (
                    publication_validation.get(
                        "within_page_limit"
                    )
                ),
                "uses_full_page_budget": (
                    uses_full_page_budget
                ),
                "references_included_in_limit": (
                    publication_validation.get(
                        "references_included_in_limit"
                    )
                ),
                "disclosure_statement_included_in_limit": (
                    publication_validation.get(
                        "disclosure_statement_included_in_limit"
                    )
                ),
                "template_manipulation_prohibited": (
                    publication_validation.get(
                        "template_manipulation_prohibited"
                    )
                ),
            }

            # Underfill revisions grow from the best valid under-limit
            # manuscript. If an expansion overshoots the page budget, compact
            # the actual over-limit candidate rather than mistakenly compacting
            # the older underfilled best candidate. This makes overshoot a useful
            # convergence step toward the exact page count.
            revision_base_manuscript = (
                revised_manuscript
                if page_count > maximum_pages
                else best_manuscript
            )

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "master_prompt": master_prompt,
                    "verified_records": (
                            manuscript_revision_context[
                                "verified_records"
                            ]
                        ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                            manuscript_revision_context[
                                "execution_manifest"
                            ]
                        ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_revision_context[
                            "manuscript_evidence_bundle"
                        ]
                    ),
                    "evidence_synthesis": (
                        manuscript_revision_context[
                            "evidence_synthesis"
                        ]
                    ),
                    "manuscript": (
                        revision_base_manuscript
                        .model_dump()
                    ),
                    "peer_review": (
                        latest_peer_review
                        .model_dump()
                    ),
                    "publication_validation": (
                        format_feedback
                    ),
                    "revision_round": (
                        "format_"
                        f"{format_round + 1}"
                    ),
                    "revision_instruction": (
                        revision_instruction
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Autonomous manuscript format revision "
                    f"{format_round + 1}"
                ),
            )

            write_json(
                revision_rounds_dir
                / (
                    "format_revised_package_"
                    f"{format_round + 1:02d}.json"
                ),
                revised_manuscript,
            )

            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

        if (
            best_publication_validation is not None
            and (
                publication_validation is None
                or best_page_count
                > (
                    publication_validation.get("page_count")
                    if isinstance(
                        publication_validation.get("page_count"),
                        int,
                    )
                    else -1
                )
            )
        ):
            revised_manuscript = best_manuscript

            # Re-render the selected best-so-far manuscript so that
            # the authoritative publication artifacts correspond to
            # the manuscript actually carried forward.
            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

        # -------------------------------------------------
        # Modular evidence-grounded exact-page convergence
        # -------------------------------------------------
        #
        # Whole-manuscript rewriting is an unreliable way to hit a
        # discrete LaTeX page boundary. If the scientifically complete
        # manuscript is still underfilled here, generate independent
        # evidence-grounded expansion variants for selected scientific
        # sections. Only the requested section is extracted from each
        # model response. Deterministic code then compiles combinations
        # and selects the richest publication-valid exact-page result.
        #
        # This is a presentation/convergence operation only:
        # - no new experiments;
        # - no new analyses;
        # - no invented evidence/statistics/references;
        # - no IEEE template manipulation.
        modular_maximum_pages = (
            publication_validation.get("maximum_pages")
            if publication_validation is not None
            else None
        )
        modular_page_count = (
            publication_validation.get("page_count")
            if publication_validation is not None
            else None
        )

        if (
            publication_validation is not None
            and publication_validation.get("compile_status")
            == "passed"
            and isinstance(modular_page_count, int)
            and isinstance(modular_maximum_pages, int)
            and modular_page_count < modular_maximum_pages
        ):
            modular_dir = (
                run_dir
                / "manuscript"
                / "modular_page_search"
            )
            modular_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Use the strongest under-limit manuscript already selected
            # by the ordinary format loop as the immutable base.
            modular_base_manuscript = revised_manuscript
            modular_base_dump = (
                modular_base_manuscript.model_dump()
            )

            modular_sections = (
                "methodology",
                "results",
                "discussion",
                "related_work",
            )

            # 0 means retain the base section.
            # 1 and 2 are independently generated medium/full variants.
            modular_variants: dict[
                str,
                dict[int, str],
            ] = {}

            modular_generation_report: dict[str, Any] = {
                "base_page_count": modular_page_count,
                "maximum_pages": modular_maximum_pages,
                "sections": {},
            }

            for modular_section in modular_sections:
                base_section_text = str(
                    modular_base_dump[
                        "sections"
                    ][modular_section]
                )
                base_section_words = len(
                    base_section_text.split()
                )

                modular_variants[modular_section] = {
                    0: base_section_text,
                }

                modular_generation_report[
                    "sections"
                ][modular_section] = {
                    "base_words": base_section_words,
                    "variants": {},
                }

                for modular_variant_level in (1, 2):
                    if modular_variant_level == 1:
                        modular_variant_instruction = (
                            "Expand ONLY the "
                            f"{modular_section} section with a moderate "
                            "amount of distinct, substantive scientific "
                            "detail supported by the supplied archived "
                            "evidence and completed analysis. Preserve the "
                            "scientific meaning and all existing supported "
                            "content in that section. Aim for roughly "
                            "1.35 to 1.55 times the current section's "
                            "substantive content. Do not modify the abstract, "
                            "other sections, references, disclosure, title, "
                            "figures, tables, empirical findings, or claims. "
                            "Do not add filler or repetition. Do not invent "
                            "experiments, data, statistics, citations, "
                            "implementation facts, or examples."
                        )
                    else:
                        modular_variant_instruction = (
                            "Expand ONLY the "
                            f"{modular_section} section substantially with "
                            "distinct technical material supported by the "
                            "supplied archived evidence and completed "
                            "analysis. Preserve all existing supported "
                            "content in that section. Aim for roughly "
                            "1.65 to 1.90 times the current section's "
                            "substantive content. Prefer concrete methodology, "
                            "execution semantics, quantitative interpretation, "
                            "artifact-grounded diagnostics, reproducibility "
                            "details, threats to validity, operational "
                            "implications, or verified related-work detail "
                            "appropriate specifically to this section. "
                            "Do not modify the abstract, other sections, "
                            "references, disclosure, title, figures, tables, "
                            "empirical findings, or claims. Do not add filler "
                            "or repetition and do not invent evidence."
                        )

                    generated_variant = await run_agent(
                        MANUSCRIPT_REVISER,
                        {
                            "master_prompt": master_prompt,
                            "verified_records": (
                            manuscript_revision_context[
                                "verified_records"
                            ]
                        ),
                            "evidence_verification": (
                                evidence_report
                            ),
                            "preregistration": (
                                preregistration.model_dump()
                            ),
                            "execution_manifest": (
                            manuscript_revision_context[
                                "execution_manifest"
                            ]
                        ),
                            "analysis_results": (
                                analysis_results
                            ),
                            "deterministic_reconciliation": (
                                deterministic_reconciliation
                            ),
                            "manuscript_evidence_bundle": (
                                manuscript_revision_context[
                                    "manuscript_evidence_bundle"
                                ]
                            ),
                            "evidence_synthesis": (
                                manuscript_revision_context[
                                    "evidence_synthesis"
                                ]
                            ),
                            "manuscript": (
                                modular_base_manuscript
                                .model_dump()
                            ),
                            "peer_review": (
                                latest_peer_review.model_dump()
                            ),
                            "publication_validation": (
                                publication_validation
                            ),
                            "revision_round": (
                                "modular_page_search_"
                                f"{modular_section}_"
                                f"{modular_variant_level}"
                            ),
                            "revision_instruction": (
                                modular_variant_instruction
                            ),
                        },
                        expected_type=ManuscriptPackage,
                        stage_name=(
                            "Modular manuscript expansion "
                            f"{modular_section} "
                            f"variant {modular_variant_level}"
                        ),
                    )

                    write_json(
                        modular_dir
                        / (
                            f"{modular_section}_"
                            f"variant_{modular_variant_level}.json"
                        ),
                        generated_variant,
                    )

                    generated_dump = (
                        generated_variant.model_dump()
                    )
                    candidate_section_text = str(
                        generated_dump[
                            "sections"
                        ][modular_section]
                    )
                    candidate_section_words = len(
                        candidate_section_text.split()
                    )

                    accepted_variant = (
                        candidate_section_words
                        > base_section_words
                    )

                    modular_generation_report[
                        "sections"
                    ][modular_section][
                        "variants"
                    ][
                        str(modular_variant_level)
                    ] = {
                        "words": candidate_section_words,
                        "accepted": accepted_variant,
                    }

                    # Critically, ignore every other field from the
                    # model response. Only the requested section may
                    # enter the deterministic reservoir.
                    if accepted_variant:
                        modular_variants[
                            modular_section
                        ][
                            modular_variant_level
                        ] = candidate_section_text

            write_json(
                modular_dir
                / "generation_report.json",
                modular_generation_report,
            )

            # ------------------------------------------------
            # Deterministic combination search.
            #
            # With 4 sections and at most {base, medium, full},
            # this is <= 3^4 = 81 inexpensive LaTeX compilations.
            # ------------------------------------------------
            modular_level_options = [
                tuple(
                    sorted(
                        modular_variants[
                            section_name
                        ].keys()
                    )
                )
                for section_name in modular_sections
            ]

            modular_exact_candidates: list[
                tuple[
                    int,
                    ManuscriptPackage,
                    dict[str, Any],
                    tuple[int, ...],
                ]
            ] = []

            modular_best_under_manuscript = (
                modular_base_manuscript
            )
            modular_best_under_validation = dict(
                publication_validation
            )
            modular_best_under_pages = modular_page_count
            modular_best_under_words = (
                manuscript_section_word_count(
                    modular_base_manuscript
                )
            )

            modular_search_report: list[
                dict[str, Any]
            ] = []

            modular_candidate_index = 0

            for modular_levels in product(
                *modular_level_options
            ):
                # Skip the all-base combination; already compiled.
                if all(
                    level == 0
                    for level in modular_levels
                ):
                    continue

                modular_candidate_index += 1

                candidate_dump = json.loads(
                    json.dumps(
                        modular_base_dump,
                        ensure_ascii=False,
                    )
                )

                for (
                    section_name,
                    level,
                ) in zip(
                    modular_sections,
                    modular_levels,
                    strict=True,
                ):
                    candidate_dump[
                        "sections"
                    ][section_name] = (
                        modular_variants[
                            section_name
                        ][level]
                    )

                modular_candidate = (
                    ManuscriptPackage.model_validate(
                        candidate_dump
                    )
                )

                candidate_output_dir = (
                    modular_dir
                    / "compiled_candidates"
                    / (
                        f"candidate_"
                        f"{modular_candidate_index:03d}"
                    )
                )

                candidate_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            modular_candidate.model_dump()
                        ),
                        verified_records=records,
                        output_dir=candidate_output_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                candidate_pages = (
                    candidate_validation.get(
                        "page_count"
                    )
                )
                candidate_words = (
                    manuscript_section_word_count(
                        modular_candidate
                    )
                )

                modular_search_report.append(
                    {
                        "candidate_index": (
                            modular_candidate_index
                        ),
                        "levels": {
                            section_name: level
                            for section_name, level in zip(
                                modular_sections,
                                modular_levels,
                                strict=True,
                            )
                        },
                        "page_count": candidate_pages,
                        "section_words": candidate_words,
                        "passed": (
                            candidate_validation.get(
                                "passed",
                                False,
                            )
                        ),
                    }
                )

                if (
                    candidate_validation.get(
                        "passed"
                    )
                    is True
                    and candidate_pages
                    == modular_maximum_pages
                ):
                    modular_exact_candidates.append(
                        (
                            candidate_words,
                            modular_candidate,
                            dict(
                                candidate_validation
                            ),
                            modular_levels,
                        )
                    )

                elif (
                    isinstance(candidate_pages, int)
                    and candidate_pages
                    <= modular_maximum_pages
                    and (
                        candidate_pages
                        > modular_best_under_pages
                        or (
                            candidate_pages
                            == modular_best_under_pages
                            and candidate_words
                            > modular_best_under_words
                        )
                    )
                ):
                    modular_best_under_manuscript = (
                        modular_candidate
                    )
                    modular_best_under_validation = dict(
                        candidate_validation
                    )
                    modular_best_under_pages = (
                        candidate_pages
                    )
                    modular_best_under_words = (
                        candidate_words
                    )

            write_json(
                modular_dir
                / "combination_search_report.json",
                modular_search_report,
            )

            if modular_exact_candidates:
                # Among exact-page publication-valid combinations,
                # select the scientifically richest by substantive
                # section word count. The LLM does not decide which
                # combination wins.
                modular_exact_candidates.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                (
                    modular_selected_words,
                    modular_selected_manuscript,
                    modular_selected_validation,
                    modular_selected_levels,
                ) = modular_exact_candidates[0]

                revised_manuscript = (
                    modular_selected_manuscript
                )

                write_json(
                    modular_dir
                    / "selected_exact_candidate.json",
                    revised_manuscript,
                )

                write_json(
                    modular_dir
                    / "selected_exact_candidate_metadata.json",
                    {
                        "levels": {
                            section_name: level
                            for section_name, level in zip(
                                modular_sections,
                                modular_selected_levels,
                                strict=True,
                            )
                        },
                        "section_words": (
                            modular_selected_words
                        ),
                        "exact_candidate_count": len(
                            modular_exact_candidates
                        ),
                    },
                )

                # Re-render selected candidate into the authoritative
                # publication directory.
                publication_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            revised_manuscript.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                write_json(
                    publication_dir
                    / "publication_validation_modular_exact.json",
                    publication_validation,
                )

                write_json(
                    run_dir
                    / "manuscript"
                    / "revised_package.json",
                    revised_manuscript,
                )

            elif (
                modular_best_under_pages
                > modular_page_count
                or (
                    modular_best_under_pages
                    == modular_page_count
                    and modular_best_under_words
                    > manuscript_section_word_count(
                        modular_base_manuscript
                    )
                )
            ):
                # No exact-five combination was found, but never lose
                # a demonstrably stronger under-limit candidate.
                revised_manuscript = (
                    modular_best_under_manuscript
                )

                publication_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            revised_manuscript.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                write_json(
                    publication_dir
                    / "publication_validation_modular_best_under.json",
                    publication_validation,
                )

                write_json(
                    run_dir
                    / "manuscript"
                    / "revised_package.json",
                    revised_manuscript,
                )


        # -------------------------------------------------
        # Protected exact-page submission checkpoint
        # -------------------------------------------------
        #
        # Once deterministic publication validation has produced
        # an exact-page manuscript, preserve it across terminal
        # peer review. Later revisions may replace this checkpoint
        # only after they themselves compile to the exact frozen
        # page budget.
        protected_submission_manuscript: (
            ManuscriptPackage | None
        ) = None
        protected_submission_validation: (
            dict[str, Any] | None
        ) = None

        if (
            publication_validation is not None
            and publication_validation.get("passed") is True
            and publication_validation.get("page_count")
            == publication_validation.get("maximum_pages")
        ):
            protected_submission_manuscript = revised_manuscript
            protected_submission_validation = dict(
                publication_validation
            )

            write_json(
                revision_rounds_dir
                / "protected_exact_page_checkpoint.json",
                protected_submission_manuscript,
            )

            write_json(
                publication_dir
                / "publication_validation_protected_checkpoint.json",
                protected_submission_validation,
            )

        previous_terminal_review: PeerReviewReport | None = None
        # -------------------------------------------------
        # Deterministic terminal-review factual accounting
        # -------------------------------------------------
        #
        # Derive a compact set of authoritative episode-level facts
        # directly from the frozen raw execution results. This supports
        # terminal review/revision without asking an LLM to infer the
        # raw-results schema or invent artifact locators.
        #
        # This is factual accounting only, not a new statistical
        # analysis and not a modification of scientific results.
        terminal_factual_accounting: dict[str, Any] = {
            "source": "execution/raw_results.jsonl",
            "discordant_pairs": [],
        }

        raw_results_path = (
            run_dir / "execution" / "raw_results.jsonl"
        )

        if raw_results_path.exists():
            episodes_by_pair: dict[
                str,
                list[dict[str, Any]],
            ] = {}

            with raw_results_path.open(
                "r",
                encoding="utf-8",
            ) as raw_results_file:
                for record_index, raw_line in enumerate(
                    raw_results_file,
                    start=1,
                ):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        raw_record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(raw_record, dict):
                        continue

                    pair_id = raw_record.get("pair_id")
                    condition = raw_record.get("condition")
                    score = raw_record.get("score")

                    if (
                        not isinstance(pair_id, str)
                        or condition
                        not in {"baseline", "guarded"}
                        or not isinstance(
                            score,
                            (int, float),
                        )
                    ):
                        continue

                    episode_fact = {
                        "raw_results_record_index": (
                            record_index
                        ),
                        "pair_id": pair_id,
                        "task_id": raw_record.get(
                            "task_id"
                        ),
                        "episode_id": raw_record.get(
                            "episode_id"
                        ),
                        "condition": condition,
                        "score": score,
                        "score_reason_code": (
                            raw_record.get(
                                "score_reason_code"
                            )
                        ),
                        "attempt_count": raw_record.get(
                            "attempt_count"
                        ),
                        "model_calls_used": (
                            raw_record.get(
                                "model_calls_used"
                            )
                        ),
                        "transformation_id": (
                            raw_record.get(
                                "transformation_id"
                            )
                        ),
                    }

                    episodes_by_pair.setdefault(
                        pair_id,
                        [],
                    ).append(episode_fact)

            for pair_id, episode_facts in sorted(
                episodes_by_pair.items()
            ):
                baseline_facts = [
                    item
                    for item in episode_facts
                    if item.get("condition")
                    == "baseline"
                ]
                guarded_facts = [
                    item
                    for item in episode_facts
                    if item.get("condition")
                    == "guarded"
                ]

                if (
                    len(baseline_facts) != 1
                    or len(guarded_facts) != 1
                ):
                    continue

                baseline_fact = baseline_facts[0]
                guarded_fact = guarded_facts[0]

                if (
                    baseline_fact.get("score")
                    == guarded_fact.get("score")
                ):
                    continue

                terminal_factual_accounting[
                    "discordant_pairs"
                ].append(
                    {
                        "pair_id": pair_id,
                        "baseline": baseline_fact,
                        "guarded": guarded_fact,
                    }
                )

        # -------------------------------------------------
        # Deterministic task-difficulty accounting
        # -------------------------------------------------
        task_manifest_path = (
            run_dir / "execution" / "task_manifest.jsonl"
        )

        difficulty_counts: dict[str, int] = {}

        if task_manifest_path.exists():
            with task_manifest_path.open(
                "r",
                encoding="utf-8",
            ) as task_manifest_file:
                for raw_line in task_manifest_file:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        task_record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(task_record, dict):
                        continue

                    task_payload = task_record.get(
                        "task_payload",
                        {},
                    )
                    if not isinstance(task_payload, dict):
                        continue

                    difficulty = task_payload.get(
                        "difficulty",
                        {},
                    )
                    if not isinstance(difficulty, dict):
                        continue

                    level = difficulty.get("level")
                    if not isinstance(level, str) or not level:
                        continue

                    difficulty_counts[level] = (
                        difficulty_counts.get(level, 0) + 1
                    )

        difficulty_total = sum(difficulty_counts.values())

        terminal_factual_accounting[
            "difficulty_summary"
        ] = {
            "total_tasks_with_difficulty": difficulty_total,
            "levels": [
                {
                    "level": level,
                    "count": count,
                    "percentage": (
                        100.0 * count / difficulty_total
                        if difficulty_total
                        else 0.0
                    ),
                }
                for level, count in sorted(
                    difficulty_counts.items()
                )
            ],
        }

        # -------------------------------------------------
        # Full bibliographic metadata for cited records only
        # -------------------------------------------------
        terminal_bibliographic_records: list[
            dict[str, Any]
        ] = []

        cited_record_ids = {
            str(value)
            for value in (
                getattr(
                    revised_manuscript,
                    "cited_record_ids",
                    [],
                )
                or []
            )
            if value is not None
        }

        for verified_record in records:
            if hasattr(verified_record, "model_dump"):
                verified_record_dict = (
                    verified_record.model_dump()
                )
            elif isinstance(verified_record, dict):
                verified_record_dict = dict(
                    verified_record
                )
            else:
                continue

            serialized_record = json.dumps(
                verified_record_dict,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            # Keep terminal bibliographic context bounded. When
            # cited_record_ids is populated, supply only records already
            # cited by the manuscript. Never interpret an empty citation
            # set as permission to inject the entire literature corpus.
            if not cited_record_ids:
                continue

            if not any(
                cited_id in serialized_record
                for cited_id in cited_record_ids
            ):
                continue

            terminal_bibliographic_records.append(
                verified_record_dict
            )

        write_json(
            revision_rounds_dir
            / "terminal_factual_accounting.json",
            terminal_factual_accounting,
        )

        write_json(
            revision_rounds_dir
            / "terminal_bibliographic_records.json",
            terminal_bibliographic_records,
        )

        maximum_terminal_revision_rounds = 3

        for terminal_round in range(
            1,
            maximum_terminal_revision_rounds + 2,
        ):
            terminal_review_mode = (
                "full_terminal_review"
                if terminal_round == 1
                else "closure_review"
            )
            latest_peer_review = await run_agent(
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
                        _compact_execution_manifest_for_manuscript(
                            execution_manifest
                        )
                    ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_evidence_bundle
                    ),
                    "manuscript": (
                        revised_manuscript.model_dump()
                    ),
                    "review_round": (
                        maximum_peer_review_rounds
                        + terminal_round
                    ),
                    "review_mode": (
                        terminal_review_mode
                    ),
                    "terminal_factual_accounting": (
                        terminal_factual_accounting
                    ),
                    "terminal_bibliographic_records": (
                        terminal_bibliographic_records
                    ),
                    "terminal_factual_accounting_instruction": (
                        "Treat terminal_factual_accounting as "
                        "authoritative for exact pair IDs, episode "
                        "IDs, discordant scores, raw-results record "
                        "indices, attempt counts, model-call counts, "
                        "and transformation identifiers. Do not "
                        "infer or invent raw-results fields such as "
                        "stage='repair' when they are not present. "
                        "Do not require machine paths, hashes, raw "
                        "record indices, or provider-call filenames "
                        "in the scientific paper merely for audit "
                        "convenience; require the scientifically "
                        "material verified fact instead. "
                        "The difficulty_summary contains deterministic "
                        "counts and percentages for every difficulty "
                        "level actually observed in the task manifest; "
                        "do not assume or require a fixed difficulty "
                        "taxonomy. terminal_bibliographic_records "
                        "contains full verified metadata for records "
                        "already cited by the manuscript. Require "
                        "normal scholarly references rather than an "
                        "archive-bibliography placeholder."
                    ),
                    "previous_terminal_review": (
                        previous_terminal_review.model_dump()
                        if previous_terminal_review is not None
                        else None
                    ),
                },
                expected_type=PeerReviewReport,
                stage_name=(
                    "Terminal AI peer review "
                    f"round {terminal_round}"
                ),
            )

            previous_terminal_review = latest_peer_review

            write_json(
                review_rounds_dir
                / (
                    "review_terminal_"
                    f"{terminal_round:02d}.json"
                ),
                latest_peer_review,
            )

            # A clean terminal review means no further manuscript
            # revision is required.
            if (
                not latest_peer_review.critical_issues
                and not latest_peer_review.required_revisions
            ):
                break

            # At most two terminal review-driven revisions are
            # permitted. The third terminal review is therefore
            # review-only and becomes the final archived judgement.
            if (
                terminal_round
                > maximum_terminal_revision_rounds
            ):
                break

            # Terminal review remediation must be cumulative.
            #
            # The protected exact-page checkpoint is a rollback safety net,
            # not the default scientific revision base. Starting every
            # terminal round from the protected checkpoint discards review
            # fixes made in earlier rounds and can cause closure reviews to
            # request the same already-addressed changes repeatedly.
            terminal_review_revision_base_manuscript = (
                revised_manuscript
            )
            terminal_review_revision_base_validation = (
                publication_validation
            )

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "master_prompt": master_prompt,
                    "verified_records": (
                            manuscript_revision_context[
                                "verified_records"
                            ]
                        ),
                    "evidence_verification": (
                        evidence_report
                    ),
                    "preregistration": (
                        preregistration.model_dump()
                    ),
                    "execution_manifest": (
                            manuscript_revision_context[
                                "execution_manifest"
                            ]
                        ),
                    "analysis_results": (
                        analysis_results
                    ),
                    "deterministic_reconciliation": (
                        deterministic_reconciliation
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_revision_context[
                            "manuscript_evidence_bundle"
                        ]
                    ),
                    "evidence_synthesis": (
                        manuscript_revision_context[
                            "evidence_synthesis"
                        ]
                    ),
                    "manuscript": (
                        terminal_review_revision_base_manuscript
                        .model_dump()
                    ),
                    "peer_review": (
                        latest_peer_review.model_dump()
                    ),
                    "terminal_factual_accounting": (
                        terminal_factual_accounting
                    ),
                    "terminal_bibliographic_records": (
                        terminal_bibliographic_records
                    ),
                    "terminal_factual_accounting_instruction": (
                        "When a required revision concerns an "
                        "episode identifier, pair identifier, "
                        "repair event, call count, or discordant "
                        "outcome, use the exact verified values "
                        "from terminal_factual_accounting. Do not "
                        "merely tell the reader how to search an "
                        "artifact and do not invent raw-result "
                        "fields or locators. State the scientifically "
                        "relevant verified fact concisely. "
                        "When difficulty composition is requested, "
                        "use difficulty_summary and report every "
                        "observed level with exact count and "
                        "percentage; do not omit levels merely because "
                        "a reviewer did not name them. When the "
                        "References section contains an archive or "
                        "verified-bibliography placeholder, replace it "
                        "with normal in-manuscript scholarly references "
                        "using terminal_bibliographic_records. Do not "
                        "invent missing bibliographic fields."
                    ),
                    "publication_validation": (
                        terminal_review_revision_base_validation
                    ),
                    "revision_round": (
                        "terminal_"
                        f"{terminal_round}"
                    ),
                    "revision_instruction": (
                        "Resolve every required revision in the supplied "
                        "terminal peer-review report using only the supplied "
                        "archived evidence and completed analysis artifacts. "
                        "Address the listed required revisions explicitly and "
                        "preserve previously resolved reviewer requirements. "
                        "Treat the supplied manuscript as the authoritative "
                        "revision base. Make only the minimum scientific edits "
                        "needed to resolve the enumerated review issues. Preserve "
                        "all unaffected sections, paragraphs, tables, references, "
                        "results, limitations, and disclosure. Do not globally "
                        "rewrite, summarize, condense, or restructure material "
                        "that the review does not require you to change. "
                        "Raw execution artifacts may support factual accounting "
                        "such as call counts, episode identifiers, repair flags, "
                        "validator reuse, missingness, and provenance, but do "
                        "not perform or invent new post-lock statistical "
                        "analyses from raw execution data. If a requested "
                        "preregistered analysis was not completed by the "
                        "autonomous analysis stage, report it transparently as "
                        "unexecuted or unavailable and qualify any dependent "
                        "claim. When this is a required peer-review revision, "
                        "state that explicitly in Methods and/or Results, not "
                        "only in the Conclusion, limitations, or Disclosure "
                        "Statement. State that no confirmatory inference from "
                        "the unexecuted analysis is reported and identify the "
                        "scope of the analysis that was actually completed and "
                        "archived. Do not perform a new post-lock analysis "
                        "merely to satisfy the reviewer. Preserve all supported "
                        "scientific content "
                        "already present. Do not change empirical results, "
                        "invent evidence, introduce new experiments, or weaken "
                        "required disclosure. For publication-style required "
                        "revisions, normalize citations and references to "
                        "standard IEEE numbered citation form; remove or "
                        "relocate run-internal artifact paths and full hashes "
                        "from ordinary scientific prose; and condense the "
                        "Disclosure Statement to the minimum track-required "
                        "provenance content while preserving every mandatory "
                        "disclosure fact. Prefer concise scientific prose and "
                        "normal bibliographic references over audit-log-style "
                        "metadata. Maintain the exact five-page IEEE "
                        "publication requirement."
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Terminal peer-review manuscript revision "
                    f"{terminal_round}"
                ),
            )

            write_json(
                revision_rounds_dir
                / (
                    "terminal_revised_package_"
                    f"{terminal_round:02d}.json"
                ),
                revised_manuscript,
            )

            # Keep the conventional latest-manuscript alias in sync
            # with the manuscript actually being reviewed/rendered.
            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                publication_dir
                / (
                    "publication_validation_terminal_"
                    f"{terminal_round:02d}.json"
                ),
                publication_validation,
            )

            # Preserve the best valid terminal-format candidate seen so far.
            # For equal page counts, prefer the manuscript with more
            # substantive serialized content so underfill revisions cannot
            # regress by rewriting/compressing a better candidate.
            best_terminal_manuscript: ManuscriptPackage | None = None
            best_terminal_publication_validation: (
                dict[str, Any] | None
            ) = None
            best_terminal_page_count = -1
            best_terminal_size = -1

            initial_terminal_page_count = (
                publication_validation.get("page_count")
            )
            initial_terminal_maximum_pages = (
                publication_validation.get("maximum_pages")
            )

            if (
                isinstance(initial_terminal_page_count, int)
                and isinstance(initial_terminal_maximum_pages, int)
                and initial_terminal_page_count
                <= initial_terminal_maximum_pages
            ):
                best_terminal_manuscript = revised_manuscript
                best_terminal_publication_validation = dict(
                    publication_validation
                )
                best_terminal_page_count = (
                    initial_terminal_page_count
                )
                best_terminal_size = len(
                    json.dumps(
                        revised_manuscript.model_dump(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

            maximum_terminal_format_rounds = 10

            for terminal_format_round in range(
                1,
                maximum_terminal_format_rounds + 1,
            ):
                terminal_compile_status = (
                    publication_validation.get(
                        "compile_status"
                    )
                )
                terminal_page_count = (
                    publication_validation.get(
                        "page_count"
                    )
                )
                terminal_maximum_pages = (
                    publication_validation.get(
                        "maximum_pages"
                    )
                )

                if terminal_compile_status != "passed":
                    break

                if (
                    terminal_page_count
                    == terminal_maximum_pages
                ):
                    break

                if (
                    terminal_page_count is None
                    or terminal_maximum_pages is None
                ):
                    break

                if terminal_page_count < terminal_maximum_pages:
                    terminal_overshoot_target_pages = (
                        terminal_maximum_pages + 1
                    )
                    # Preserve the current review-remediated manuscript
                    # during page-format recovery. The protected exact-page
                    # checkpoint is reserved for rollback only; preferring it
                    # here would silently discard terminal-review fixes.
                    terminal_base = (
                        best_terminal_manuscript
                        if best_terminal_manuscript is not None
                        else revised_manuscript
                    )
                    terminal_current_words = (
                        manuscript_section_word_count(
                            terminal_base
                        )
                    )
                    terminal_minimum_words = max(
                        2300,
                        int(terminal_current_words * 1.45),
                    )
                    terminal_preferred_words = max(
                        2600,
                        int(terminal_current_words * 1.65),
                    )

                    terminal_format_instruction = (
                        "The manuscript remains under the frozen page budget. "
                        "Do not try to approach five pages cautiously from below. "
                        f"Deliberately target approximately "
                        f"{terminal_overshoot_target_pages} compiled IEEE pages "
                        "in this expansion step, after which the next round can "
                        "compact the actual over-limit candidate to exactly "
                        f"{terminal_maximum_pages} pages. Produce at least "
                        f"{terminal_minimum_words} substantive scientific-section "
                        "words, preferably around "
                        f"{terminal_preferred_words} where supported by the "
                        "archived evidence. The word floor is a convergence "
                        "control, not permission for filler or repetition. "
                        "Preserve every reviewer-resolved and artifact-supported "
                        "statement already present. Do not remove, shorten, "
                        "summarize, or replace supported content. Add at least "
                        "eight distinct new artifact-grounded paragraphs across "
                        "Methods, Results, statistical interpretation, execution "
                        "accounting, Discussion/Limitations, reproducibility, and "
                        "verified related work, plus compact artifact-grounded "
                        "tables or figures where supported. Every addition must "
                        "communicate distinct information from the supplied "
                        "archived evidence. Do not invent evidence, statistics, "
                        "references, experiments, or formatting tricks."
                    )

                else:
                    terminal_format_instruction = (
                        "The terminal-review revision exceeds the frozen "
                        f"{terminal_maximum_pages}-page IEEE budget. "
                        "Compact it to exactly that length while preserving "
                        "all reviewer-resolved scientific content, results, "
                        "limitations, references, and Disclosure Statement. "
                        "Do not alter empirical findings or formatting rules."
                    )

                # Monotonic terminal revision-base selection:
                #
                # * If a reviewer-resolved candidate is overfull, compact
                #   that actual overfull candidate.
                # * If a reviewer rewrite is underfull and an exact-page
                #   checkpoint exists, restart from the checkpoint and
                #   reapply the review requirements.
                # * Otherwise preserve the ordinary best-candidate
                #   convergence behaviour.
                if terminal_page_count > terminal_maximum_pages:
                    terminal_revision_base_manuscript = (
                        revised_manuscript
                    )
                    terminal_revision_base_publication_validation = (
                        publication_validation
                    )

                elif (
                    terminal_page_count < terminal_maximum_pages
                    and protected_submission_manuscript is not None
                    and protected_submission_validation is not None
                ):
                    terminal_revision_base_manuscript = (
                        protected_submission_manuscript
                    )
                    terminal_revision_base_publication_validation = (
                        protected_submission_validation
                    )

                else:
                    terminal_revision_base_manuscript = (
                        best_terminal_manuscript
                        if best_terminal_manuscript is not None
                        else revised_manuscript
                    )
                    terminal_revision_base_publication_validation = (
                        best_terminal_publication_validation
                        if best_terminal_publication_validation
                        is not None
                        else publication_validation
                    )

                revised_manuscript = await run_agent(
                    MANUSCRIPT_REVISER,
                    {
                        "master_prompt": master_prompt,
                        "verified_records": (
                            manuscript_revision_context[
                                "verified_records"
                            ]
                        ),
                        "evidence_verification": (
                            evidence_report
                        ),
                        "preregistration": (
                            preregistration.model_dump()
                        ),
                        "execution_manifest": (
                            manuscript_revision_context[
                                "execution_manifest"
                            ]
                        ),
                        "analysis_results": (
                            analysis_results
                        ),
                        "deterministic_reconciliation": (
                            deterministic_reconciliation
                        ),
                        "manuscript_evidence_bundle": (
                            manuscript_revision_context[
                                "manuscript_evidence_bundle"
                            ]
                        ),
                        "evidence_synthesis": (
                            manuscript_revision_context[
                                "evidence_synthesis"
                            ]
                        ),
                        "manuscript": (
                            terminal_revision_base_manuscript.model_dump()
                        ),
                        "peer_review": (
                            latest_peer_review.model_dump()
                        ),
                        "publication_validation": (
                            terminal_revision_base_publication_validation
                        ),
                        "previous_revision_publication_validation": (
                            publication_validation
                        ),
                        "revision_round": (
                            "terminal_format_"
                            f"{terminal_round}_"
                            f"{terminal_format_round}"
                        ),
                        "revision_instruction": (
                            terminal_format_instruction
                        ),
                    },
                    expected_type=ManuscriptPackage,
                    stage_name=(
                        "Terminal manuscript page convergence "
                        f"{terminal_round}."
                        f"{terminal_format_round}"
                    ),
                )

                write_json(
                    revision_rounds_dir
                    / (
                        "terminal_format_revised_package_"
                        f"{terminal_round:02d}_"
                        f"{terminal_format_round:02d}.json"
                    ),
                    revised_manuscript,
                )

                write_json(
                    run_dir
                    / "manuscript"
                    / "revised_package.json",
                    revised_manuscript,
                )

                publication_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            revised_manuscript.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                write_json(
                    publication_dir
                    / (
                        "publication_validation_terminal_format_"
                        f"{terminal_round:02d}_"
                        f"{terminal_format_round:02d}.json"
                    ),
                    publication_validation,
                )

                # Evaluate the newly rendered candidate. A candidate is
                # eligible only when it is within the frozen page limit.
                # Prefer higher page count; for equal page count, prefer
                # greater manuscript content.
                current_terminal_page_count = (
                    publication_validation.get(
                        "page_count"
                    )
                )
                current_terminal_maximum_pages = (
                    publication_validation.get(
                        "maximum_pages"
                    )
                )
                current_terminal_size = len(
                    json.dumps(
                        revised_manuscript.model_dump(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

                current_terminal_candidate_is_better = False

                if (
                    isinstance(
                        current_terminal_page_count,
                        int,
                    )
                    and isinstance(
                        current_terminal_maximum_pages,
                        int,
                    )
                    and current_terminal_page_count
                    <= current_terminal_maximum_pages
                ):
                    if (
                        current_terminal_page_count
                        > best_terminal_page_count
                    ):
                        current_terminal_candidate_is_better = True

                    elif (
                        current_terminal_page_count
                        == best_terminal_page_count
                        and current_terminal_size
                        > best_terminal_size
                    ):
                        current_terminal_candidate_is_better = True

                if current_terminal_candidate_is_better:
                    best_terminal_manuscript = revised_manuscript
                    best_terminal_publication_validation = dict(
                        publication_validation
                    )
                    best_terminal_page_count = (
                        current_terminal_page_count
                    )
                    best_terminal_size = current_terminal_size

                # If this attempt has reached the exact frozen page
                # budget, no further format-model call is necessary.
                if (
                    publication_validation.get("passed") is True
                    and publication_validation.get(
                        "compile_status"
                    )
                    == "passed"
                    and current_terminal_page_count
                    == current_terminal_maximum_pages
                ):
                    break

            # -------------------------------------------------
            # Monotonic exact-page terminal selection
            # -------------------------------------------------
            selected_exact_terminal_candidate = (
                best_terminal_manuscript is not None
                and best_terminal_publication_validation is not None
                and best_terminal_publication_validation.get(
                    "passed"
                )
                is True
                and best_terminal_publication_validation.get(
                    "compile_status"
                )
                == "passed"
                and best_terminal_publication_validation.get(
                    "page_count"
                )
                == best_terminal_publication_validation.get(
                    "maximum_pages"
                )
            )

            if selected_exact_terminal_candidate:
                # A reviewer-revised manuscript reached exact-page
                # validity. Promote it to the new protected checkpoint.
                revised_manuscript = best_terminal_manuscript
                publication_validation = dict(
                    best_terminal_publication_validation
                )

                protected_submission_manuscript = revised_manuscript
                protected_submission_validation = dict(
                    publication_validation
                )

                write_json(
                    revision_rounds_dir
                    / (
                        "protected_exact_page_checkpoint_"
                        f"terminal_{terminal_round:02d}.json"
                    ),
                    protected_submission_manuscript,
                )

                write_json(
                    publication_dir
                    / (
                        "publication_validation_protected_checkpoint_"
                        f"terminal_{terminal_round:02d}.json"
                    ),
                    protected_submission_validation,
                )

            elif (
                protected_submission_manuscript is not None
                and protected_submission_validation is not None
            ):
                # The terminal rewrite/convergence did not preserve the
                # exact page budget. Restore the last submission-valid
                # manuscript rather than regressing to a shorter paper.
                #
                # The current peer-review report remains unchanged, so
                # unresolved scientific requirements remain visible to
                # the next closure review/final judge.
                revised_manuscript = (
                    protected_submission_manuscript
                )
                publication_validation = dict(
                    protected_submission_validation
                )

            elif (
                best_terminal_manuscript is not None
                and best_terminal_publication_validation is not None
            ):
                # No pre-existing exact-page checkpoint: retain the
                # legacy best-candidate behaviour.
                revised_manuscript = best_terminal_manuscript
                publication_validation = dict(
                    best_terminal_publication_validation
                )

            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

            # Re-render exactly the manuscript that is actually carried
            # forward. Authoritative PDF/TeX artifacts can therefore
            # never correspond to a discarded terminal candidate.
            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                publication_dir
                / (
                    "publication_validation_terminal_best_"
                    f"{terminal_round:02d}.json"
                ),
                publication_validation,
            )

        # Compatibility/latest-terminal-review alias.
        write_json(
            review_rounds_dir
            / "review_terminal.json",
            latest_peer_review,
        )

        if publication_validation is None:
            raise RuntimeError(
                "Publication validation did not execute."
            )

        # Compatibility/latest-publication-validation alias.
        write_json(
            publication_dir
            / "publication_validation.json",
            publication_validation,
        )

        # -------------------------------------------------
        # 15. Final autonomous readiness judgement
        # -------------------------------------------------

        # -----------------------------------------------------
        # Deterministic final publication-sanity gate.
        #
        # This is deliberately non-scientific: it checks only
        # typesetting/layout and machine-generated manuscript
        # hygiene. Scientific outcomes are never modified here.
        # -----------------------------------------------------
        # -----------------------------------------------------
        # Deterministic final manuscript hygiene/provenance audit
        # with one bounded autonomous remediation opportunity.
        #
        # The remediation is publication/infrastructure-only:
        # scientific design, execution, results, statistics, and
        # supported conclusions are frozen.
        # -----------------------------------------------------
        publication_sanity_audit = (
            audit_manuscript_publication_sanity(
                run_dir=run_dir,
            )
        )

        artifact_reference_audit = (
            audit_manuscript_artifact_references(
                manuscript=revised_manuscript,
                run_dir=run_dir,
            )
        )

        write_json(
            publication_dir
            / "publication_sanity_audit_pre_remediation.json",
            publication_sanity_audit,
        )
        write_json(
            publication_dir
            / "artifact_reference_audit_pre_remediation.json",
            artifact_reference_audit,
        )

        manuscript_hygiene_needs_remediation = (
            publication_sanity_audit.get("passed") is not True
            or artifact_reference_audit.get("passed") is not True
        )

        if manuscript_hygiene_needs_remediation:
            remediation_instruction = (
                "Perform one bounded final publication-hygiene and provenance "
                "repair of the supplied manuscript. The deterministic audits "
                "listed below are authoritative. Correct only those reported "
                "publication, typesetting, metadata-pollution, and artifact-"
                "reference defects. "
                "\n\n"
                "SCIENTIFIC CONTENT IS FROZEN. Do not change the research "
                "question, preregistration, experimental design, execution, "
                "sample sizes, numerical results, statistical tests, effect "
                "sizes, confidence intervals, p-values, supported scientific "
                "interpretation, limitations, scientific conclusions, or "
                "verified citation set. Do not perform or invent any new "
                "analysis, experiment, model call, datum, statistic, citation, "
                "artifact, path, hash, DOI, repository location, or result. "
                "\n\n"
                "Rewrite machine-oriented material as normal scientific "
                "conference prose, but treat this strictly as a CONTENT-PRESERVING "
                "hygiene transformation, not as summarization, compression, or "
                "copy-editing for brevity. Do not merely delete offending metadata. "
                "For every affected passage, remove the machine locator, path, hash, "
                "command, audit-language, or provenance wrapper while preserving the "
                "supported scientific explanation carried by that passage. "
                "\n\n"
                "Do not collapse several substantive sentences into one shorter "
                "sentence merely because metadata is being removed. Preserve the "
                "existing explanatory depth, paragraph structure, section balance, "
                "and approximately the same rendered scientific volume wherever the "
                "underlying material is scientifically meaningful. The input "
                "manuscript already reflects deliberate convergence toward the exact "
                "five-page IEEE budget; hygiene repair must not turn a full scientific "
                "paper into an underfilled one by deleting supported exposition. "
                "\n\n"
                "Where an artifact inventory or filesystem reference itself has no "
                "scientific value, remove that machine-oriented element, but retain "
                "or restore the surrounding supported methodology, result "
                "interpretation, comparison, limitation, uncertainty, failure-mode, "
                "or operational implication in ordinary prose. If removal of "
                "machine-oriented material would materially shorten a section, use "
                "the supplied verified evidence and completed analysis to express "
                "the SAME already-supported scientific substance more explicitly in "
                "that section. Do not introduce any new claim, datum, experiment, "
                "analysis, citation, result, or interpretation. "
                "\n\n"
                "Related-work prose should synthesize and compare prior findings "
                "with normal citations; methodology prose should explain design, "
                "measurement, and validation choices; results prose should report "
                "and interpret completed outcomes; discussion prose should explain "
                "implications, failure modes, uncertainty, limitations, and "
                "operational relevance. Preserve all unaffected manuscript material "
                "verbatim in substance. Do not use generic filler, vague transitions, "
                "provenance boilerplate, metadata restatements, or formatting tricks "
                "to maintain page count. "
                "\n\n"
                "Remove unnecessary artifact inventories and filesystem paths. "
                "In the Disclosure Statement, do not include a repository path "
                "merely to establish that the required master prompt was archived; "
                "state that provenance fact in normal prose and retain the required "
                "immutable master-prompt SHA-256. Keep detailed machine provenance "
                "in the archived run rather than copying it into the paper. Remove "
                "full SHA-256 digests except the one immutable master-prompt digest "
                "required in the mandatory Disclosure Statement. Do not print DOI "
                "metadata in body prose. Remove raw reproduction commands and "
                "reviewer-response/meta-review language. "
                "\n\n"
                "If an artifact/path/hash claim is unsupported, remove or rephrase "
                "that provenance claim; never invent a replacement. When a real "
                "scientific fact is already supported, state the scientific fact "
                "directly rather than substituting an artifact pointer. "
                "\n\n"
                "Repair any reported LaTeX column or margin overflow by scientific "
                "rewriting and removal of machine metadata without unnecessarily "
                "compressing substantive content. Do not use smaller fonts, margin "
                "changes, spacing tricks, geometry changes, or other formatting hacks. "
                "Preserve normal IEEE formatting and the exact five-page publication "
                "requirement. "
                "\n\n"
                "Make the minimum edits needed to satisfy the deterministic audits. "
                "Preserve all unaffected manuscript material. The desired result is "
                "the same scientifically substantive manuscript with machine-oriented "
                "presentation defects removed, still occupying exactly the required "
                "five IEEE pages. Do not trade scientific completeness for metadata "
                "cleanliness."
            )

            revised_manuscript = await run_agent(
                MANUSCRIPT_REVISER,
                {
                    "current_manuscript": (
                        revised_manuscript.model_dump()
                    ),

                    # Use the same deterministic bounded evidence
                    # context as every other manuscript revision.
                    # This gives the hygiene remediation enough
                    # verified provenance/scientific context to
                    # preserve factual content without reintroducing
                    # the r46 context-overflow problem.
                    "verified_records": (
                        manuscript_revision_context[
                            "verified_records"
                        ]
                    ),
                    "execution_manifest": (
                        manuscript_revision_context[
                            "execution_manifest"
                        ]
                    ),
                    "manuscript_evidence_bundle": (
                        manuscript_revision_context[
                            "manuscript_evidence_bundle"
                        ]
                    ),
                    "evidence_synthesis": (
                        manuscript_revision_context[
                            "evidence_synthesis"
                        ]
                    ),

                    "publication_validation": (
                        publication_validation
                    ),
                    "publication_sanity_audit": (
                        publication_sanity_audit
                    ),
                    "artifact_reference_audit": (
                        artifact_reference_audit
                    ),
                    "paper_run_constraints": (
                        paper_run_constraints
                    ),
                    "revision_mode": (
                        "deterministic_publication_hygiene_remediation"
                    ),
                    "revision_instruction": (
                        remediation_instruction
                    ),
                },
                expected_type=ManuscriptPackage,
                stage_name=(
                    "Deterministic publication hygiene remediation"
                ),
            )

            write_json(
                revision_rounds_dir
                / "publication_hygiene_remediated_package.json",
                revised_manuscript,
            )

            # Keep the conventional authoritative manuscript alias
            # synchronized with the manuscript actually carried forward.
            write_json(
                run_dir
                / "manuscript"
                / "revised_package.json",
                revised_manuscript,
            )

            # Re-render the remediated manuscript. The resulting PDF/TeX
            # becomes authoritative only for this post-audit candidate.
            publication_validation = (
                build_publication_artifacts(
                    manuscript=(
                        revised_manuscript.model_dump()
                    ),
                    verified_records=records,
                    output_dir=publication_dir,
                    paper_run_constraints=(
                        paper_run_constraints
                    ),
                )
            )

            write_json(
                publication_dir
                / "publication_validation_post_hygiene_remediation.json",
                publication_validation,
            )

            # -------------------------------------------------
            # Deterministic protected seed for post-hygiene recovery
            # -------------------------------------------------
            #
            # A protected exact-page checkpoint may contain more supported
            # scientific substance than the current post-hygiene candidate.
            # Publication sanitation can reduce that protected checkpoint
            # below the exact page budget. Probe it deterministically here,
            # BEFORE the existing bounded scientific recovery loop, so the
            # existing recovery machinery may use the stronger clean seed.
            #
            # No scientific agent call occurs in this seed-selection step.

            if (
                protected_submission_manuscript is not None
                and protected_submission_validation is not None
                and protected_submission_validation.get("passed") is True
                and protected_submission_validation.get("page_count")
                == protected_submission_validation.get("maximum_pages")
            ):
                pre_seed_manuscript = revised_manuscript
                pre_seed_validation = dict(
                    publication_validation
                    if publication_validation is not None
                    else {}
                )

                sanitized_protected_recovery_seed = (
                    sanitize_structured_manuscript_publication_metadata(
                        protected_submission_manuscript,
                        run_dir=run_dir,
                    )
                )

                write_json(
                    revision_rounds_dir
                    / "post_hygiene_protected_recovery_seed.json",
                    sanitized_protected_recovery_seed,
                )

                protected_seed_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            sanitized_protected_recovery_seed.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                protected_seed_sanity = (
                    audit_manuscript_publication_sanity(
                        run_dir=run_dir,
                    )
                )

                protected_seed_artifact_audit = (
                    audit_manuscript_artifact_references(
                        manuscript=(
                            sanitized_protected_recovery_seed
                        ),
                        run_dir=run_dir,
                    )
                )

                write_json(
                    publication_dir
                    / (
                        "publication_validation_"
                        "post_hygiene_protected_recovery_seed.json"
                    ),
                    protected_seed_validation,
                )
                write_json(
                    publication_dir
                    / (
                        "publication_sanity_audit_"
                        "post_hygiene_protected_recovery_seed.json"
                    ),
                    protected_seed_sanity,
                )
                write_json(
                    publication_dir
                    / (
                        "artifact_reference_audit_"
                        "post_hygiene_protected_recovery_seed.json"
                    ),
                    protected_seed_artifact_audit,
                )

                current_seed_page_count = (
                    pre_seed_validation.get("page_count")
                )
                protected_seed_page_count = (
                    protected_seed_validation.get("page_count")
                )
                protected_seed_maximum_pages = (
                    protected_seed_validation.get("maximum_pages")
                )

                current_seed_text_length = len(
                    _manuscript_text(
                        pre_seed_manuscript
                    )
                )
                protected_seed_text_length = len(
                    _manuscript_text(
                        sanitized_protected_recovery_seed
                    )
                )

                protected_seed_is_clean_and_useful = (
                    protected_seed_validation.get("compile_status")
                    == "passed"
                    and protected_seed_sanity.get("passed") is True
                    and protected_seed_artifact_audit.get("passed")
                    is True
                    and isinstance(protected_seed_page_count, int)
                    and isinstance(protected_seed_maximum_pages, int)
                    and protected_seed_page_count
                    <= protected_seed_maximum_pages
                    and (
                        not isinstance(current_seed_page_count, int)
                        or protected_seed_page_count
                        >= current_seed_page_count
                    )
                    and protected_seed_text_length
                    > current_seed_text_length
                )

                if protected_seed_is_clean_and_useful:
                    revised_manuscript = (
                        sanitized_protected_recovery_seed
                    )
                    publication_validation = dict(
                        protected_seed_validation
                    )

                    write_json(
                        revision_rounds_dir
                        / (
                            "selected_post_hygiene_"
                            "protected_recovery_seed.json"
                        ),
                        revised_manuscript,
                    )

                    write_json(
                        run_dir
                        / "manuscript"
                        / "revised_package.json",
                        revised_manuscript,
                    )

                else:
                    # The deterministic probe is non-destructive. Restore
                    # the manuscript that entered seed selection and rerender
                    # it so authoritative PDF/TeX artifacts remain aligned.
                    revised_manuscript = pre_seed_manuscript
                    publication_validation = (
                        build_publication_artifacts(
                            manuscript=(
                                revised_manuscript.model_dump()
                            ),
                            verified_records=records,
                            output_dir=publication_dir,
                            paper_run_constraints=(
                                paper_run_constraints
                            ),
                        )
                    )

            # -------------------------------------------------
            # Monotonic bounded post-hygiene scientific
            # underfill recovery
            # -------------------------------------------------
            #
            # Hygiene can remove non-scientific material from an
            # exact-page manuscript. Recovery is allowed only by
            # elaborating scientific content already supported by the
            # frozen evidence.
            #
            # Recovery is MONOTONIC:
            # - every attempt starts from the best successfully compiled
            #   manuscript seen so far;
            # - a failed compile is discarded;
            # - a candidate that does not increase page count is discarded;
            # - an overshooting candidate is discarded;
            # - an exact-page candidate is accepted immediately.
            #
            # Therefore a later failed/shrinking revision can never replace
            # an earlier better manuscript.

            post_hygiene_page_count = (
                publication_validation.get("page_count")
                if publication_validation is not None
                else None
            )
            post_hygiene_maximum_pages = (
                publication_validation.get("maximum_pages")
                if publication_validation is not None
                else None
            )

            if (
                publication_validation is not None
                and publication_validation.get("compile_status") == "passed"
                and isinstance(post_hygiene_page_count, int)
                and isinstance(post_hygiene_maximum_pages, int)
                and post_hygiene_page_count < post_hygiene_maximum_pages
            ):
                maximum_post_hygiene_underfill_attempts = 12

                best_recovery_manuscript = revised_manuscript
                best_recovery_validation = publication_validation
                best_recovery_page_count = post_hygiene_page_count

                for underfill_attempt in range(
                    1,
                    maximum_post_hygiene_underfill_attempts + 1,
                ):
                    if (
                        best_recovery_page_count
                        == post_hygiene_maximum_pages
                    ):
                        break

                    scientific_underfill_instruction = (
                        "The manuscript is scientifically frozen but "
                        "underfills the required IEEE page budget after "
                        "publication-hygiene cleanup. "
                        f"This is bounded monotonic recovery attempt "
                        f"{underfill_attempt} of "
                        f"{maximum_post_hygiene_underfill_attempts}. "
                        f"The best manuscript currently occupies "
                        f"{best_recovery_page_count} page(s) of "
                        f"{post_hygiene_maximum_pages}. "
                        "\n\n"
                        "Page count is a coarse rendered measure. A useful "
                        "scientific expansion may still compile to the same "
                        "integer page count. Build CUMULATIVELY on the supplied "
                        "manuscript: preserve every supported scientific "
                        "expansion already present and add further supported "
                        "substance. Do not rewrite a four-page manuscript into "
                        "a different four-page manuscript of similar or smaller "
                        "scientific volume. "
                        "\n\n"
                        "SCIENTIFIC CONTENT IS FROZEN. Use ONLY material "
                        "already supported by the supplied verified "
                        "literature synthesis, preregistration, completed "
                        "methodology, execution evidence, analysis results, "
                        "deterministic reconciliation, and existing "
                        "manuscript evidence. "
                        "\n\n"
                        "Preserve the existing substantive scientific prose. "
                        "Do not shorten, delete, or replace supported material "
                        "merely for stylistic rewriting. "
                        "\n\n"
                        "The authoritative target is EXACTLY the required "
                        "five compiled IEEE pages. The immediately preceding "
                        "publication-hygiene step removed non-scientific "
                        "material from a manuscript that previously reached "
                        "the exact page budget. Restore the lost volume ONLY "
                        "with scientifically substantive, evidence-grounded "
                        "content. Do not pad, repeat provenance, manipulate "
                        "formatting, enlarge headings, or add generic filler. "
                        "\n\n"
                        "For this recovery attempt, make MULTIPLE cumulative "
                        "substantive additions rather than one small rewrite. "
                        "Where supported by the supplied frozen evidence, "
                        "expand several of the following in the same revision: "
                        "(1) related-work synthesis and the precise gap; "
                        "(2) experimental-design rationale and why the paired "
                        "shared-candidate estimand is appropriate; "
                        "(3) interpretation of the observed effect and "
                        "discordant-pair structure; "
                        "(4) difficulty-stratified or workflow-pattern "
                        "interpretation already present in execution evidence; "
                        "(5) failure-mode analysis grounded in archived "
                        "validator/repair outcomes; "
                        "(6) uncertainty, statistical power, and ceiling/floor "
                        "implications supported by the completed analysis; "
                        "(7) threats to internal/external validity; and "
                        "(8) qualified operational implications and concrete "
                        "future validation questions. "
                        "\n\n"
                        "Do not merely paraphrase existing sentences. Add "
                        "distinct scientific reasoning, comparisons, or "
                        "evidence-backed interpretation. If the supplied "
                        "evidence supports a compact scientific table that "
                        "clarifies results, difficulty composition, or failure "
                        "modes, you may add it; do not invent measurements. "
                        "\n\n"
                        "Add further supported "
                        "scientific explanation where useful. Prefer deeper "
                        "literature synthesis and comparison, methodology and "
                        "design rationale, interpretation of existing "
                        "quantitative results, observed failure modes and "
                        "uncertainty, threats to validity and limitations, "
                        "and operational implications for reliable LLM/agent "
                        "NetOps. "
                        "\n\n"
                        "Do not change or introduce the research question, "
                        "hypotheses, preregistration, study design, execution, "
                        "sample sizes, numerical results, statistical tests, "
                        "effect sizes, confidence intervals, p-values, "
                        "supported interpretations, scientific conclusions, "
                        "citations, evidence records, experiments, analyses, "
                        "model calls, data, or results. "
                        "\n\n"
                        "Do not add generic filler, repetition, provenance "
                        "boilerplate, artifact inventories, filesystem paths, "
                        "hashes, DOI labels, raw commands, reviewer-response "
                        "language, or template/spacing tricks. "
                        "\n\n"
                        "The purpose of this attempt is specifically to add "
                        "genuine supported scientific substance without "
                        "removing existing supported substance, so that the "
                        "compiled manuscript moves closer to exactly the "
                        "frozen IEEE page budget."
                    )

                    candidate_recovery_manuscript = await run_agent(
                        MANUSCRIPT_REVISER,
                        {
                            "current_manuscript": (
                                best_recovery_manuscript.model_dump()
                            ),
                            "verified_records": (
                                manuscript_revision_context[
                                    "verified_records"
                                ]
                            ),
                            "execution_manifest": (
                                manuscript_revision_context[
                                    "execution_manifest"
                                ]
                            ),
                            "manuscript_evidence_bundle": (
                                manuscript_revision_context[
                                    "manuscript_evidence_bundle"
                                ]
                            ),
                            "evidence_synthesis": (
                                manuscript_revision_context[
                                    "evidence_synthesis"
                                ]
                            ),
                            "preregistration": (
                                preregistration.model_dump()
                            ),
                            "analysis_results": analysis_results,
                            "deterministic_reconciliation": (
                                deterministic_reconciliation
                            ),
                            "publication_validation": (
                                best_recovery_validation
                            ),
                            "paper_run_constraints": (
                                paper_run_constraints
                            ),
                            "revision_mode": (
                                "post_hygiene_scientific_"
                                "underfill_recovery"
                            ),
                            "revision_instruction": (
                                scientific_underfill_instruction
                            ),
                        },
                        expected_type=ManuscriptPackage,
                        stage_name=(
                            "Post-hygiene scientific underfill recovery "
                            f"attempt {underfill_attempt}"
                        ),
                    )

                    write_json(
                        revision_rounds_dir
                        / (
                            "post_hygiene_underfill_candidate_"
                            f"{underfill_attempt:02d}.json"
                        ),
                        candidate_recovery_manuscript,
                    )

                    candidate_validation = (
                        build_publication_artifacts(
                            manuscript=(
                                candidate_recovery_manuscript.model_dump()
                            ),
                            verified_records=records,
                            output_dir=publication_dir,
                            paper_run_constraints=(
                                paper_run_constraints
                            ),
                        )
                    )

                    write_json(
                        publication_dir
                        / (
                            "publication_validation_"
                            "post_hygiene_underfill_recovery_"
                            f"{underfill_attempt:02d}.json"
                        ),
                        candidate_validation,
                    )

                    # A recovery candidate must improve page count
                    # AND remain publication-safe. A longer candidate
                    # that reintroduces provenance pollution, broken
                    # references, missing glyphs, overflow, or bogus
                    # artifact claims is not an improvement.
                    candidate_publication_sanity = (
                        audit_manuscript_publication_sanity(
                            run_dir=run_dir,
                        )
                    )

                    candidate_artifact_reference_audit = (
                        audit_manuscript_artifact_references(
                            manuscript=(
                                candidate_recovery_manuscript
                            ),
                            run_dir=run_dir,
                        )
                    )

                    write_json(
                        publication_dir
                        / (
                            "publication_sanity_audit_"
                            "post_hygiene_underfill_"
                            f"{underfill_attempt:02d}.json"
                        ),
                        candidate_publication_sanity,
                    )

                    write_json(
                        publication_dir
                        / (
                            "artifact_reference_audit_"
                            "post_hygiene_underfill_"
                            f"{underfill_attempt:02d}.json"
                        ),
                        candidate_artifact_reference_audit,
                    )

                    candidate_page_count = (
                        candidate_validation.get("page_count")
                    )

                    current_recovery_text_length = len(
                        _manuscript_text(
                            best_recovery_manuscript
                        )
                    )

                    candidate_recovery_text_length = len(
                        _manuscript_text(
                            candidate_recovery_manuscript
                        )
                    )

                    candidate_is_clean_and_non_regressing = (
                        candidate_validation.get("compile_status")
                        == "passed"
                        and candidate_publication_sanity.get(
                            "passed"
                        )
                        is True
                        and candidate_artifact_reference_audit.get(
                            "passed"
                        )
                        is True
                        and isinstance(candidate_page_count, int)
                        and candidate_page_count
                        <= post_hygiene_maximum_pages
                        and candidate_page_count
                        >= best_recovery_page_count
                        and candidate_recovery_text_length
                        > current_recovery_text_length
                    )

                    if candidate_is_clean_and_non_regressing:
                        best_recovery_manuscript = (
                            candidate_recovery_manuscript
                        )
                        best_recovery_validation = (
                            candidate_validation
                        )
                        best_recovery_page_count = (
                            candidate_page_count
                        )

                        write_json(
                            revision_rounds_dir
                            / (
                                "post_hygiene_underfill_"
                                "recovered_package.json"
                            ),
                            best_recovery_manuscript,
                        )

                        write_json(
                            run_dir
                            / "manuscript"
                            / "revised_package.json",
                            best_recovery_manuscript,
                        )

                    if (
                        best_recovery_page_count
                        == post_hygiene_maximum_pages
                    ):
                        break

                # Restore/render the best candidate seen across ALL attempts.
                revised_manuscript = best_recovery_manuscript
                publication_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            revised_manuscript.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                write_json(
                    revision_rounds_dir
                    / "post_hygiene_underfill_recovered_package.json",
                    revised_manuscript,
                )

                write_json(
                    run_dir
                    / "manuscript"
                    / "revised_package.json",
                    revised_manuscript,
                )

                write_json(
                    publication_dir
                    / (
                        "publication_validation_"
                        "post_hygiene_underfill_recovery.json"
                    ),
                    publication_validation,
                )

            # -------------------------------------------------
            # Deterministic protected-candidate rescue
            # -------------------------------------------------
            #
            # Late publication-hygiene/remediation stages must never
            # destroy an earlier submission-valid exact-page state.
            #
            # If the current candidate is no longer publication-valid,
            # independently re-render the latest protected exact-page
            # checkpoint and subject it to the SAME deterministic
            # publication-sanity and artifact-reference audits.
            #
            # No scientific generation or revision occurs here.
            # The protected candidate is selected only if ALL hard
            # publication constraints still pass. If the probe fails,
            # the current candidate is restored exactly.

            current_candidate_needs_rescue = (
                publication_validation is None
                or publication_validation.get("passed") is not True
                or publication_validation.get("page_count")
                != publication_validation.get("maximum_pages")
            )

            if (
                current_candidate_needs_rescue
                and protected_submission_manuscript is not None
                and protected_submission_validation is not None
                and protected_submission_validation.get("passed") is True
                and protected_submission_validation.get("page_count")
                == protected_submission_validation.get("maximum_pages")
            ):
                pre_rescue_manuscript = revised_manuscript
                pre_rescue_validation = dict(
                    publication_validation
                    if publication_validation is not None
                    else {}
                )

                sanitized_protected_submission_manuscript = (
                    sanitize_structured_manuscript_publication_metadata(
                        protected_submission_manuscript,
                        run_dir=run_dir,
                    )
                )

                write_json(
                    revision_rounds_dir
                    / "protected_final_sanitized_package.json",
                    sanitized_protected_submission_manuscript,
                )

                protected_rescue_validation = (
                    build_publication_artifacts(
                        manuscript=(
                            sanitized_protected_submission_manuscript.model_dump()
                        ),
                        verified_records=records,
                        output_dir=publication_dir,
                        paper_run_constraints=(
                            paper_run_constraints
                        ),
                    )
                )

                protected_rescue_sanity = (
                    audit_manuscript_publication_sanity(
                        run_dir=run_dir,
                    )
                )

                protected_rescue_artifact_audit = (
                    audit_manuscript_artifact_references(
                        manuscript=(
                            sanitized_protected_submission_manuscript
                        ),
                        run_dir=run_dir,
                    )
                )

                write_json(
                    publication_dir
                    / (
                        "publication_validation_"
                        "protected_final_rescue.json"
                    ),
                    protected_rescue_validation,
                )

                write_json(
                    publication_dir
                    / (
                        "publication_sanity_audit_"
                        "protected_final_rescue.json"
                    ),
                    protected_rescue_sanity,
                )

                write_json(
                    publication_dir
                    / (
                        "artifact_reference_audit_"
                        "protected_final_rescue.json"
                    ),
                    protected_rescue_artifact_audit,
                )

                protected_rescue_is_valid = (
                    protected_rescue_validation.get("passed") is True
                    and protected_rescue_validation.get(
                        "compile_status"
                    )
                    == "passed"
                    and protected_rescue_validation.get("page_count")
                    == protected_rescue_validation.get(
                        "maximum_pages"
                    )
                    and protected_rescue_sanity.get("passed") is True
                    and protected_rescue_artifact_audit.get(
                        "passed"
                    )
                    is True
                )

                if protected_rescue_is_valid:
                    revised_manuscript = (
                        sanitized_protected_submission_manuscript
                    )
                    publication_validation = dict(
                        protected_rescue_validation
                    )

                    write_json(
                        revision_rounds_dir
                        / (
                            "selected_protected_final_"
                            "candidate.json"
                        ),
                        revised_manuscript,
                    )

                    write_json(
                        run_dir
                        / "manuscript"
                        / "revised_package.json",
                        revised_manuscript,
                    )

                else:
                    # The protected checkpoint was exact-page when
                    # originally created but does not satisfy the current
                    # complete deterministic gate. Restore the candidate
                    # that entered this probe so candidate evaluation is
                    # non-destructive.
                    revised_manuscript = pre_rescue_manuscript

                    publication_validation = (
                        build_publication_artifacts(
                            manuscript=(
                                revised_manuscript.model_dump()
                            ),
                            verified_records=records,
                            output_dir=publication_dir,
                            paper_run_constraints=(
                                paper_run_constraints
                            ),
                        )
                    )

                    # Preserve the prior validation snapshot for
                    # provenance/debugging; authoritative validation is
                    # the freshly rendered restored candidate above.
                    write_json(
                        publication_dir
                        / (
                            "publication_validation_"
                            "pre_protected_final_rescue.json"
                        ),
                        pre_rescue_validation,
                    )

            # Re-run BOTH deterministic audits on the final
            # post-remediation candidate, including any bounded
            # scientific underfill recovery.
            publication_sanity_audit = (
                audit_manuscript_publication_sanity(
                    run_dir=run_dir,
                )
            )

            artifact_reference_audit = (
                audit_manuscript_artifact_references(
                    manuscript=revised_manuscript,
                    run_dir=run_dir,
                )
            )

            write_json(
                publication_dir
                / "publication_sanity_audit_post_remediation.json",
                publication_sanity_audit,
            )
            write_json(
                publication_dir
                / "artifact_reference_audit_post_remediation.json",
                artifact_reference_audit,
            )

        # Final authoritative publication checkpoint.
        revised_manuscript = sanitize_structured_manuscript_publication_metadata(
            revised_manuscript,
            run_dir=run_dir,
        )

        write_json(
            run_dir / "manuscript" / "revised_package.json",
            revised_manuscript,
        )

        publication_validation = build_publication_artifacts(
            manuscript=revised_manuscript.model_dump(),
            verified_records=records,
            output_dir=publication_dir,
            paper_run_constraints=paper_run_constraints,
        )

        publication_sanity_audit = audit_manuscript_publication_sanity(
            run_dir=run_dir,
        )

        artifact_reference_audit = audit_manuscript_artifact_references(
            manuscript=revised_manuscript,
            run_dir=run_dir,
        )

        write_json(
            publication_dir / "publication_validation_final_authoritative.json",
            publication_validation,
        )
        write_json(
            publication_dir / "publication_sanity_audit_final_authoritative.json",
            publication_sanity_audit,
        )
        write_json(
            publication_dir / "artifact_reference_audit_final_authoritative.json",
            artifact_reference_audit,
        )

        # Compatibility/final aliases always describe the manuscript
        # that is actually entering the final readiness decision.
        write_json(
            publication_dir
            / "publication_sanity_audit.json",
            publication_sanity_audit,
        )
        write_json(
            publication_dir
            / "artifact_reference_audit.json",
            artifact_reference_audit,
        )
        write_json(
            publication_dir
            / "publication_validation.json",
            publication_validation,
        )

        final_deterministic_gate_failures: list[str] = []

        # Exact five compiled pages is a hard submission invariant.
        # A manuscript that merely compiles within the maximum is not
        # submission-ready for this paper run.
        if (
            publication_validation.get("compile_status") != "passed"
            or publication_validation.get("page_count")
            != publication_validation.get("maximum_pages")
        ):
            final_deterministic_gate_failures.append(
                "Final manuscript is not an exact-page submission: "
                f"compile_status={publication_validation.get('compile_status')}, "
                f"page_count={publication_validation.get('page_count')}, "
                f"required_pages={publication_validation.get('maximum_pages')}."
            )

        if publication_validation.get("passed") is not True:
            final_deterministic_gate_failures.append(
                "Final manuscript publication validation failed after "
                "publication-hygiene remediation."
            )

        if publication_sanity_audit.get("passed") is not True:
            final_deterministic_gate_failures.append(
                "Deterministic manuscript publication-sanity audit failed."
            )

        if artifact_reference_audit.get("passed") is not True:
            final_deterministic_gate_failures.append(
                "Deterministic manuscript artifact-reference audit failed."
            )

        if final_deterministic_gate_failures:
            final_deterministic_warnings = list(
                publication_sanity_audit.get(
                    "issues",
                    [],
                )
            )
            final_deterministic_warnings.extend(
                artifact_reference_audit.get(
                    "issues",
                    [],
                )
            )

            if publication_validation.get("passed") is not True:
                final_deterministic_warnings.append(
                    "Post-remediation publication validation did not pass; "
                    f"page_count={publication_validation.get('page_count')}, "
                    f"maximum_pages={publication_validation.get('maximum_pages')}, "
                    f"compile_status={publication_validation.get('compile_status')}."
                )

            final_deterministic_warnings = sorted(
                set(final_deterministic_warnings)
            )

            final_report = FinalReadinessReport(
                ready=False,
                passed_gates=[
                    "Autonomous research execution completed",
                    "Final manuscript compilation completed",
                ],
                failed_gates=(
                    final_deterministic_gate_failures
                ),
                warnings=(
                    final_deterministic_warnings
                ),
                final_state=(
                    "MANUSCRIPT_FINAL_DETERMINISTIC_AUDIT_FAILED"
                ),
            )

            write_json(
                run_dir
                / "final_readiness_report.json",
                final_report.model_dump(
                    mode="json"
                ),
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
                    "ready": False,
                    "publication_sanity_audit": (
                        publication_sanity_audit
                    ),
                    "artifact_reference_audit": (
                        artifact_reference_audit
                    ),
                    "publication_validation": (
                        publication_validation
                    ),
                },
            )

            return final_report

        # -----------------------------------------------------
        # Final authoritative peer-review closure.
        #
        # Earlier terminal reviews necessarily inspect manuscript states
        # that can subsequently be changed by deterministic publication
        # hygiene, exact-page recovery, or protected-candidate rescue.
        # Re-review exactly the authoritative manuscript that passed the
        # final deterministic gates so stale objections are neither carried
        # forward nor silently overridden.
        # -----------------------------------------------------
        final_closure_review = await run_agent(
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
                    _compact_execution_manifest_for_manuscript(
                        execution_manifest
                    )
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
                "manuscript_evidence_bundle": (
                    manuscript_evidence_bundle
                ),
                "manuscript": (
                    revised_manuscript.model_dump()
                ),
                "review_round": (
                    maximum_peer_review_rounds
                    + maximum_terminal_revision_rounds
                    + 2
                ),
                "review_mode": (
                    "final_authoritative_closure_review"
                ),
                "terminal_factual_accounting": (
                    terminal_factual_accounting
                ),
                "terminal_bibliographic_records": (
                    terminal_bibliographic_records
                ),
                "publication_validation": (
                    publication_validation
                ),
                "publication_sanity_audit": (
                    publication_sanity_audit
                ),
                "artifact_reference_audit": (
                    artifact_reference_audit
                ),
                "terminal_factual_accounting_instruction": (
                    "Treat terminal_factual_accounting as "
                    "authoritative for exact pair IDs, episode "
                    "IDs, discordant scores, raw-results record "
                    "indices, attempt counts, model-call counts, "
                    "and transformation identifiers. Do not "
                    "infer or invent raw-results fields. Do not "
                    "require machine paths, hashes, raw record "
                    "indices, or provider-call filenames in the "
                    "scientific paper merely for audit "
                    "convenience; require the scientifically "
                    "material verified fact instead. "
                    "terminal_bibliographic_records contains "
                    "verified metadata for records already cited "
                    "by the manuscript. Require normal scholarly "
                    "references and verify that cited records "
                    "support the proximate claims."
                ),
                "previous_terminal_review": (
                    latest_peer_review.model_dump()
                    if latest_peer_review is not None
                    else None
                ),
                "final_authoritative_closure_instruction": (
                    "This is the final peer-review closure of the "
                    "authoritative manuscript that has already "
                    "passed the final deterministic publication, "
                    "sanity, artifact-reference, and exact-page "
                    "gates. Judge ONLY the current manuscript "
                    "supplied in this request. The previous "
                    "terminal review is historical context, not "
                    "an automatically inherited set of failures. "
                    "For every previous objection, verify whether "
                    "the defect is still present in the current "
                    "manuscript before repeating it. In "
                    "particular, use publication_sanity_audit, "
                    "artifact_reference_audit, and "
                    "publication_validation as authoritative for "
                    "current formatting/provenance properties "
                    "such as full hashes, artifact paths, raw DOI "
                    "citation forms, compilation, and page count. "
                    "Do not require a revision for an earlier "
                    "formatting or provenance defect that those "
                    "final artifacts show has been removed. "
                    "The single archived master-prompt locator "
                    "provenance/master_prompt.txt and its associated "
                    "full SHA-256 are a mandatory Disclosure exception. "
                    "When the final sanity audit reports exactly one "
                    "artifact path and exactly one full SHA-256 and they "
                    "are that master-prompt reference, do not reject or "
                    "request removal of them as provenance clutter. "
                    "A preregistration identifier must be stated, but a "
                    "preregistration-manifest SHA-256 is explicitly NOT "
                    "required in the manuscript. Its digest belongs in "
                    "the archived provenance bundle. Do not request "
                    "insertion of a preregistration hash or treat its "
                    "absence as an unresolved revision. "
                    "When the current Methods states that "
                    "temperature=null means unset, does not imply "
                    "temperature=0 or deterministic model sampling, and "
                    "records one sampled generation reused under "
                    "shared_initial_candidate semantics, treat that issue "
                    "as resolved. When Related Work has weakened an "
                    "effectiveness claim to the level actually supported "
                    "by the verified bibliographic records, treat that "
                    "citation-semantic issue as resolved. "
                    "However, retain any genuine unresolved "
                    "scientific, methodological, accounting, "
                    "citation-semantic, or disclosure-content "
                    "problem that is still present in the current "
                    "manuscript. This stage is review-only: do not "
                    "request new experiments merely to improve an "
                    "unfavorable scientific result."
                ),
            },
            expected_type=PeerReviewReport,
            stage_name=(
                "Final authoritative AI peer-review closure"
            ),
        )

        write_json(
            review_rounds_dir
            / "review_final_closure.json",
            final_closure_review,
        )

        # From this point onward the authoritative peer-review state must
        # correspond to exactly the manuscript entering final judgement.
        latest_peer_review = final_closure_review

        # Keep the compatibility alias aligned with the true latest review.
        write_json(
            review_rounds_dir
            / "review_terminal.json",
            latest_peer_review,
        )

        final_report = await run_agent(
            FINAL_JUDGE,
            {
                "master_prompt": master_prompt,
                "capability_manifest": (
                    capability_manifest
                ),
                "paper_run_constraints": (
                    paper_run_constraints
                ),
                "publication_validation": (
                    publication_validation
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
                    _compact_execution_manifest_for_manuscript(
                        execution_manifest
                    )
                ),
                "analysis_results": (
                    analysis_results
                ),
                "deterministic_reconciliation": (
                    deterministic_reconciliation
                ),
                "manuscript": (
                    revised_manuscript
                    .model_dump()
                ),
                "peer_review": (
                    latest_peer_review
                    .model_dump()
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
