from cnsm_agentic.autonomous_research.evidence_verification import (
    verify_evidence,
)


def test_missing_is_critical() -> None:
    result = verify_evidence(
        records=[
            {
                "record_id": "r1",
                "title": "P",
                "abstract": "A",
                "publication_year": 2025,
                "doi": "10/x",
                "url": None,
            }
        ],
        synthesis={
            "established_findings": [
                {
                    "evidence_record_ids": [
                        "r1",
                        "missing",
                    ]
                }
            ]
        },
        candidates={
            "candidates": []
        },
        decision={
            "evidence_record_ids": []
        },
    )

    assert result.missing_record_ids == [
        "missing"
    ]
    assert result.critical_issues


def test_complete_scores_one() -> None:
    result = verify_evidence(
        records=[
            {
                "record_id": "r1",
                "title": "P",
                "abstract": "A",
                "publication_year": 2025,
                "doi": "10/x",
                "url": None,
            }
        ],
        synthesis={
            "established_findings": [
                {
                    "evidence_record_ids": [
                        "r1"
                    ]
                }
            ]
        },
        candidates={
            "candidates": []
        },
        decision={
            "evidence_record_ids": []
        },
    )

    assert result.quality_score == 1.0
    assert result.missing_record_ids == []
    assert result.critical_issues == []


def test_bare_doi_resolves_to_doi_prefixed_record_id() -> None:
    result = verify_evidence(
        records=[
            {
                "record_id": (
                    "doi:10.1002/nem.2313"
                ),
                "title": (
                    "Intent-Based Network Configuration "
                    "Using Large Language Models"
                ),
                "abstract": (
                    "A test abstract."
                ),
                "publication_year": 2024,
                "doi": (
                    "10.1002/nem.2313"
                ),
                "url": (
                    "https://doi.org/"
                    "10.1002/nem.2313"
                ),
            }
        ],
        synthesis={
            "established_findings": [
                {
                    "evidence_record_ids": [
                        "10.1002/nem.2313"
                    ]
                }
            ],
            "unresolved_questions": [],
            "candidate_gaps": [],
        },
        candidates={
            "candidates": []
        },
        decision={
            "evidence_record_ids": []
        },
    )

    assert result.missing_record_ids == []
    assert result.critical_issues == []
    assert result.referenced_record_count == 1
    assert result.quality_score == 1.0


def test_doi_url_resolves_to_bare_doi_record() -> None:
    result = verify_evidence(
        records=[
            {
                "record_id": (
                    "doi:10.2139/ssrn.6365350"
                ),
                "title": "FANCY-X",
                "abstract": (
                    "A test abstract."
                ),
                "publication_year": 2026,
                "doi": (
                    "10.2139/ssrn.6365350"
                ),
                "url": (
                    "https://doi.org/"
                    "10.2139/ssrn.6365350"
                ),
            }
        ],
        synthesis={
            "established_findings": [
                {
                    "evidence_record_ids": [
                        "https://doi.org/"
                        "10.2139/ssrn.6365350"
                    ]
                }
            ],
            "unresolved_questions": [],
            "candidate_gaps": [],
        },
        candidates={
            "candidates": []
        },
        decision={
            "evidence_record_ids": []
        },
    )

    assert result.missing_record_ids == []
    assert result.critical_issues == []
    assert result.quality_score == 1.0


def test_openalex_identifier_still_resolves() -> None:
    result = verify_evidence(
        records=[
            {
                "record_id": (
                    "https://openalex.org/W4399629579"
                ),
                "title": "OpenAlex paper",
                "abstract": (
                    "A test abstract."
                ),
                "publication_year": 2025,
                "doi": None,
                "url": (
                    "https://openalex.org/W4399629579"
                ),
            }
        ],
        synthesis={
            "established_findings": [
                {
                    "evidence_record_ids": [
                        "https://openalex.org/W4399629579"
                    ]
                }
            ],
            "unresolved_questions": [],
            "candidate_gaps": [],
        },
        candidates={
            "candidates": []
        },
        decision={
            "evidence_record_ids": []
        },
    )

    assert result.missing_record_ids == []
    assert result.critical_issues == []
    assert result.quality_score == 1.0
