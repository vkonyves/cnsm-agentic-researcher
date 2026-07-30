from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


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

    normalised = str(value).strip().lower()

    if not normalised:
        return None

    doi_prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )

    for prefix in doi_prefixes:
        if normalised.startswith(prefix):
            normalised = normalised[
                len(prefix):
            ]
            break

    return normalised.rstrip("/")


def _doi(
    value: str | None,
) -> str | None:
    normalised = normalise_evidence_id(
        value
    )

    if (
        normalised is None
        or not normalised.startswith("10.")
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
        value = record.get(field)

        normalised = normalise_evidence_id(
            value
        )

        if normalised:
            aliases.add(
                normalised
            )

    doi = _doi(
        record.get("doi")
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
    Build an identifier-to-record lookup containing DOI and
    OpenAlex aliases.

    If two records expose the same alias, the first retrieved
    record is retained. Duplicate DOI reporting is handled
    independently.
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


def collect_referenced_ids(
    *,
    synthesis: dict[str, Any],
    candidates: dict[str, Any],
    decision: dict[str, Any],
) -> set[str]:
    references: set[str] = set()

    for section in (
        "established_findings",
        "unresolved_questions",
        "candidate_gaps",
    ):
        for claim in synthesis.get(
            section,
            [],
        ):
            references.update(
                claim.get(
                    "evidence_record_ids",
                    [],
                )
            )

    for candidate in candidates.get(
        "candidates",
        [],
    ):
        references.update(
            candidate.get(
                "novelty_evidence_ids",
                [],
            )
        )
        references.update(
            candidate.get(
                "feasibility_evidence_ids",
                [],
            )
        )

    references.update(
        decision.get(
            "evidence_record_ids",
            [],
        )
    )

    return {
        str(reference).strip()
        for reference in references
        if str(reference).strip()
    }


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

    alias_index = build_evidence_alias_index(
        records
    )

    resolved_records: dict[
        str,
        dict[str, Any],
    ] = {}

    missing: list[str] = []

    for original_reference in sorted(
        references
    ):
        canonical_reference = (
            normalise_evidence_id(
                original_reference
            )
        )

        if (
            canonical_reference is None
            or canonical_reference
            not in alias_index
        ):
            missing.append(
                original_reference
            )
            continue

        resolved_records[
            original_reference
        ] = alias_index[
            canonical_reference
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
            record.get("doi")
            or record.get("url")
            or record.get("openalex_id")
            or record.get("record_id")
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
                record.get("doi")
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

    if missing:
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
        len(references),
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
        missing_record_ids=missing,
        metadata_incomplete_record_ids=(
            incomplete
        ),
        metadata_only_record_ids=(
            metadata_only
        ),
        duplicate_dois=duplicate_dois,
        quality_score=quality_score,
        critical_issues=critical_issues,
        warnings=warnings,
    )
