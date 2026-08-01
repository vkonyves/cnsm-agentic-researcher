from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


CLAIM_ID_PATTERN = re.compile(
    r"^(?:EF|CG|UQ)\d+$",
    flags=re.IGNORECASE,
)


SYNTHESIS_CLAIM_SECTIONS = (
    "established_findings",
    "unresolved_questions",
    "candidate_gaps",
)


@dataclass(frozen=True)
class EvidenceVerificationResult:
    total_records: int
    referenced_record_count: int
    missing_record_ids: list[str]
    metadata_incomplete_record_ids: list[str]
    metadata_only_record_ids: list[str]
    duplicate_dois: list[str]
    quality_score: float
    critical_issues: list[str]
    warnings: list[str]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return self.__dict__.copy()


def normalise_evidence_id(
    value: str | None,
) -> str | None:
    """
    Convert equivalent DOI representations to one canonical form.

    Examples that resolve identically:
    - 10.1002/nem.2313
    - doi:10.1002/nem.2313
    - https://doi.org/10.1002/nem.2313

    OpenAlex and other non-DOI identifiers are lower-cased and
    stripped of trailing slashes but otherwise preserved.
    """
    if value is None:
        return None

    normalised = str(
        value
    ).strip().lower()

    if not normalised:
        return None

    # Recover a narrowly defined malformed hybrid identifier observed in
    # generated synthesis output, for example:
    #
    # https://openalex.org/W36227/techrxiv.173386065.57486944/v1
    #
    # The embedded W36227 corresponds to the DOI registrant prefix
    # 10.36227, while the remaining path is the TechRxiv DOI suffix.
    # Restrict recovery to TechRxiv-shaped suffixes so legitimate
    # OpenAlex work URLs are never rewritten.
    malformed_techrxiv_match = re.fullmatch(
        r"https?://openalex\.org/w(\d{5})/"
        r"(techrxiv\.[^?#\s]+)",
        normalised,
        flags=re.IGNORECASE,
    )

    if malformed_techrxiv_match:
        doi_registrant = malformed_techrxiv_match.group(1)
        doi_suffix = malformed_techrxiv_match.group(2)

        normalised = (
            f"10.{doi_registrant}/"
            f"{doi_suffix}"
        )

    doi_prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )

    for prefix in doi_prefixes:
        if normalised.startswith(
            prefix
        ):
            normalised = normalised[
                len(prefix):
            ]
            break

    return normalised.rstrip(
        "/"
    )


def _doi(
    value: str | None,
) -> str | None:
    normalised = normalise_evidence_id(
        value
    )

    if (
        normalised is None
        or not normalised.startswith(
            "10."
        )
    ):
        return None

    return normalised


def record_aliases(
    record: dict[str, Any],
) -> set[str]:
    """
    Return all identifiers by which a retrieved record may be cited.
    """
    aliases: set[str] = set()

    for field in (
        "record_id",
        "id",
        "openalex_id",
        "doi",
        "url",
    ):
        value = record.get(
            field
        )

        normalised = normalise_evidence_id(
            value
        )

        if normalised:
            aliases.add(
                normalised
            )

    doi = _doi(
        record.get(
            "doi"
        )
    )

    if doi:
        aliases.add(
            doi
        )

    return aliases


def build_evidence_alias_index(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build an identifier-to-record lookup containing DOI, URL,
    record-ID, and OpenAlex aliases.

    If two records expose the same alias, the first retrieved record
    is retained. Duplicate DOI reporting is handled independently.
    """
    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        for alias in record_aliases(
            record
        ):
            index.setdefault(
                alias,
                record,
            )

    return index

def recover_uniquely_truncated_openalex_id(
    reference: str,
    *,
    alias_index: dict[
        str,
        dict[str, Any],
    ],
) -> str | None:
    """
    Recover an OpenAlex work identifier missing exactly one trailing digit.

    Recovery is allowed only when:

    - the unresolved reference is an OpenAlex work URL;
    - a registered OpenAlex alias has exactly one additional digit;
    - that registered identifier starts with the requested identifier;
    - exactly one unique matching identifier exists.

    No recovery is attempted for ambiguous prefixes or other identifier
    families.
    """
    canonical_reference = (
        normalise_evidence_id(
            reference
        )
    )

    if canonical_reference is None:
        return None

    requested_match = re.fullmatch(
        r"https://openalex\.org/w(\d+)",
        canonical_reference,
        flags=re.IGNORECASE,
    )

    if requested_match is None:
        return None

    requested_digits = (
        requested_match.group(1)
    )

    candidates: set[str] = set()

    for alias in alias_index:
        alias_match = re.fullmatch(
            r"https://openalex\.org/w(\d+)",
            alias,
            flags=re.IGNORECASE,
        )

        if alias_match is None:
            continue

        alias_digits = (
            alias_match.group(1)
        )

        if (
            len(alias_digits)
            == len(requested_digits) + 1
            and alias_digits.startswith(
                requested_digits
            )
        ):
            candidates.add(
                alias
            )

    if len(candidates) != 1:
        return None

    return next(
        iter(
            candidates
        )
    )


def _string_list(
    value: Any,
) -> list[str]:
    """
    Convert a possible list of identifiers to cleaned strings.
    """
    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def build_claim_evidence_index(
    synthesis: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Map synthesis claim IDs such as EF1, UQ3, and CG2 to the
    bibliographic record IDs supporting those claims.

    Claim IDs are internal links between synthesis, candidate
    generation, and selection. They are not bibliographic IDs and
    must be resolved before evidence verification.
    """
    index: dict[
        str,
        list[str],
    ] = {}

    for section_name in SYNTHESIS_CLAIM_SECTIONS:
        claims = synthesis.get(
            section_name,
            [],
        )

        if not isinstance(
            claims,
            list,
        ):
            continue

        for claim in claims:
            if not isinstance(
                claim,
                dict,
            ):
                continue

            claim_id = str(
                claim.get(
                    "claim_id",
                    "",
                )
            ).strip()

            if not claim_id:
                continue

            evidence_ids = _string_list(
                claim.get(
                    "evidence_record_ids",
                    [],
                )
            )

            index[
                claim_id.upper()
            ] = evidence_ids

    return index


def resolve_evidence_references(
    *,
    references: list[str] | set[str],
    claim_evidence_index: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    """
    Resolve mixed evidence references.

    References may be:
    - synthesis claim IDs such as EF1 or CG2;
    - direct DOI, OpenAlex, URL, or record identifiers.

    Returns:
        resolved bibliographic identifiers;
        unresolved synthesis claim identifiers.

    An unknown value matching the synthesis-claim format is treated as
    an unresolved claim rather than as a bibliographic record ID.
    """
    resolved: set[str] = set()
    unresolved_claim_ids: set[str] = set()

    pending = [
        str(reference).strip()
        for reference in references
        if str(reference).strip()
    ]

    visited_claim_ids: set[str] = set()

    while pending:
        reference = pending.pop(
            0
        )

        claim_key = reference.upper()

        if claim_key in claim_evidence_index:
            if claim_key in visited_claim_ids:
                continue

            visited_claim_ids.add(
                claim_key
            )

            supporting_ids = (
                claim_evidence_index[
                    claim_key
                ]
            )

            if not supporting_ids:
                unresolved_claim_ids.add(
                    reference
                )
                continue

            pending.extend(
                supporting_ids
            )
            continue

        if CLAIM_ID_PATTERN.fullmatch(
            reference
        ):
            unresolved_claim_ids.add(
                reference
            )
            continue

        resolved.add(
            reference
        )

    return (
        resolved,
        unresolved_claim_ids,
    )


def _selected_candidates(
    *,
    candidates: dict[str, Any],
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return only the selected candidate when its identity is available.

    Falling back to all candidates preserves compatibility with older
    development artifacts that do not record selected_candidate_id.
    """
    candidate_items = candidates.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidate_items,
        list,
    ):
        return []

    valid_candidates = [
        candidate
        for candidate in candidate_items
        if isinstance(
            candidate,
            dict,
        )
    ]

    selected_candidate_id = str(
        decision.get(
            "selected_candidate_id",
            "",
        )
    ).strip()

    if not selected_candidate_id:
        return valid_candidates

    selected: list[
        dict[str, Any]
    ] = []

    for candidate in valid_candidates:
        candidate_id = str(
            candidate.get(
                "candidate_id",
                candidate.get(
                    "id",
                    "",
                ),
            )
        ).strip()

        if (
            candidate_id
            == selected_candidate_id
        ):
            selected.append(
                candidate
            )

    # A missing selected candidate is a separate candidate-identity
    # problem. Do not silently verify all rejected candidates instead.
    return selected


def collect_referenced_ids(
    *,
    synthesis: dict[str, Any],
    candidates: dict[str, Any],
    decision: dict[str, Any],
) -> set[str]:
    """
    Collect bibliographic identifiers relevant to evidence verification.

    This includes:
    - evidence supporting synthesis claims;
    - evidence referenced by the selected candidate;
    - evidence referenced by the final selection decision.

    Candidate and decision references may contain synthesis claim IDs.
    These are resolved to their supporting bibliographic identifiers.
    """
    claim_evidence_index = (
        build_claim_evidence_index(
            synthesis
        )
    )

    direct_synthesis_references: set[
        str
    ] = set()

    for section_name in SYNTHESIS_CLAIM_SECTIONS:
        claims = synthesis.get(
            section_name,
            [],
        )

        if not isinstance(
            claims,
            list,
        ):
            continue

        for claim in claims:
            if not isinstance(
                claim,
                dict,
            ):
                continue

            direct_synthesis_references.update(
                _string_list(
                    claim.get(
                        "evidence_record_ids",
                        [],
                    )
                )
            )

    candidate_references: set[
        str
    ] = set()

    for candidate in _selected_candidates(
        candidates=candidates,
        decision=decision,
    ):
        candidate_references.update(
            _string_list(
                candidate.get(
                    "novelty_evidence_ids",
                    [],
                )
            )
        )

        candidate_references.update(
            _string_list(
                candidate.get(
                    "feasibility_evidence_ids",
                    [],
                )
            )
        )

    decision_references = set(
        _string_list(
            decision.get(
                "evidence_record_ids",
                [],
            )
        )
    )

    mixed_references = (
        direct_synthesis_references
        | candidate_references
        | decision_references
    )

    resolved, unresolved_claim_ids = (
        resolve_evidence_references(
            references=mixed_references,
            claim_evidence_index=(
                claim_evidence_index
            ),
        )
    )

    # Preserve unresolved claim IDs so verify_evidence records them as
    # missing and produces a controlled evidence-repair stop.
    return (
        resolved
        | unresolved_claim_ids
    )


def verify_evidence(
    *,
    records: list[dict[str, Any]],
    synthesis: dict[str, Any],
    candidates: dict[str, Any],
    decision: dict[str, Any],
) -> EvidenceVerificationResult:
    references = collect_referenced_ids(
        synthesis=synthesis,
        candidates=candidates,
        decision=decision,
    )

    alias_index = (
        build_evidence_alias_index(
            records
        )
    )

    resolved_records: dict[
        str,
        dict[str, Any],
    ] = {}

    missing: list[str] = []

    recovered_references: dict[
        str,
        str,
    ] = {}

    for original_reference in sorted(
        references
    ):
        canonical_reference = (
            normalise_evidence_id(
                original_reference
            )
        )

        resolved_reference: str | None = None

        if (
            canonical_reference is not None
            and canonical_reference
            in alias_index
        ):
            resolved_reference = (
                canonical_reference
            )
        elif canonical_reference is not None:
            resolved_reference = (
                recover_uniquely_truncated_openalex_id(
                    original_reference,
                    alias_index=alias_index,
                )
            )

            if resolved_reference is not None:
                recovered_references[
                    original_reference
                ] = resolved_reference

        if resolved_reference is None:
            missing.append(
                original_reference
            )
            continue

        resolved_records[
            original_reference
        ] = alias_index[
            resolved_reference
        ]

    incomplete: list[str] = []
    metadata_only: list[str] = []
    abstracts = 0

    for original_reference in sorted(
        resolved_records
    ):
        record = resolved_records[
            original_reference
        ]

        title_present = bool(
            str(
                record.get(
                    "title",
                    "",
                )
            ).strip()
        )

        year_present = (
            record.get(
                "publication_year"
            )
            is not None
        )

        identity_present = bool(
            record.get(
                "doi"
            )
            or record.get(
                "url"
            )
            or record.get(
                "openalex_id"
            )
            or record.get(
                "record_id"
            )
        )

        if not (
            title_present
            and year_present
            and identity_present
        ):
            incomplete.append(
                original_reference
            )

        if record.get(
            "abstract"
        ):
            abstracts += 1
        else:
            metadata_only.append(
                original_reference
            )

    doi_counts = Counter(
        doi
        for doi in (
            _doi(
                record.get(
                    "doi"
                )
            )
            for record in records
        )
        if doi
    )

    duplicate_dois = sorted(
        doi
        for doi, count
        in doi_counts.items()
        if count > 1
    )

    critical_issues: list[str] = []
    warnings: list[str] = []

    for original_reference, recovered_reference in sorted(
        recovered_references.items()
    ):
        warnings.append(
            "Recovered uniquely truncated OpenAlex "
            "identifier: "
            f"{original_reference} -> "
            f"{recovered_reference}"
        )

    missing_claim_ids = sorted(
        reference
        for reference in missing
        if CLAIM_ID_PATTERN.fullmatch(
            reference
        )
    )

    missing_bibliographic_ids = sorted(
        reference
        for reference in missing
        if not CLAIM_ID_PATTERN.fullmatch(
            reference
        )
    )

    if missing_claim_ids:
        critical_issues.append(
            "Synthesis claim IDs do not resolve to supporting "
            "bibliographic records."
        )

    if missing_bibliographic_ids:
        critical_issues.append(
            "Evidence IDs do not resolve to retrieved records."
        )

    if incomplete:
        critical_issues.append(
            "Referenced records lack complete bibliographic identity."
        )

    if references and abstracts == 0:
        critical_issues.append(
            "No referenced record contains an abstract."
        )

    if metadata_only:
        warnings.append(
            "Some support is metadata-only and needs deeper "
            "verification in the final run."
        )

    if duplicate_dois:
        warnings.append(
            "Duplicate DOI records remain."
        )

    denominator = max(
        len(
            references
        ),
        1,
    )

    quality_score = round(
        0.40
        * (
            len(references)
            - len(missing)
        )
        / denominator
        + 0.35
        * (
            len(references)
            - len(incomplete)
        )
        / denominator
        + 0.25
        * abstracts
        / denominator,
        6,
    )

    return EvidenceVerificationResult(
        total_records=len(
            records
        ),
        referenced_record_count=len(
            references
        ),
        missing_record_ids=sorted(
            missing
        ),
        metadata_incomplete_record_ids=(
            sorted(
                incomplete
            )
        ),
        metadata_only_record_ids=(
            sorted(
                metadata_only
            )
        ),
        duplicate_dois=duplicate_dois,
        quality_score=quality_score,
        critical_issues=critical_issues,
        warnings=warnings,
    )