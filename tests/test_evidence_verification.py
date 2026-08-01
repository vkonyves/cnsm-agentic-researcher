from cnsm_agentic.autonomous_research.evidence_verification import (
    build_claim_evidence_index,
    build_evidence_alias_index,
    collect_referenced_ids,
    normalise_evidence_id,
    resolve_evidence_references,
    recover_uniquely_truncated_openalex_id,
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
    

def test_repaired_design_bare_doi_resolves_against_prefixed_record() -> None:
    records = [
        {
            "record_id": (
                "doi:10.2139/ssrn.6740060"
            ),
            "doi": (
                "10.2139/ssrn.6740060"
            ),
            "title": (
                "Prompt Injection and Jailbreak Attacks"
            ),
            "publication_year": 2026,
            "url": (
                "https://doi.org/"
                "10.2139/ssrn.6740060"
            ),
            "abstract": "A test abstract.",
        }
    ]

    alias_index = build_evidence_alias_index(
        records
    )

    cited_id = (
        "10.2139/ssrn.6740060"
    )

    assert (
        normalise_evidence_id(
            cited_id
        )
        in alias_index
    )      
    
    
def test_claim_ids_resolve_to_supporting_records() -> None:
    synthesis = {
        "established_findings": [
            {
                "claim_id": "EF1",
                "evidence_record_ids": [
                    "https://openalex.org/W123",
                ],
            }
        ],
        "unresolved_questions": [
            {
                "claim_id": "UQ3",
                "evidence_record_ids": [
                    "doi:10.1234/example",
                ],
            }
        ],
        "candidate_gaps": [
            {
                "claim_id": "CG1",
                "evidence_record_ids": [
                    "https://openalex.org/W456",
                ],
            }
        ],
    }

    claim_index = build_claim_evidence_index(
        synthesis
    )

    resolved, unresolved = (
        resolve_evidence_references(
            references={
                "EF1",
                "UQ3",
                "CG1",
            },
            claim_evidence_index=claim_index,
        )
    )

    assert resolved == {
        "https://openalex.org/W123",
        "doi:10.1234/example",
        "https://openalex.org/W456",
    }

    assert unresolved == set()


def test_unknown_claim_id_remains_unresolved() -> None:
    resolved, unresolved = (
        resolve_evidence_references(
            references={
                "CG99",
            },
            claim_evidence_index={
                "CG1": [
                    "https://openalex.org/W123",
                ]
            },
        )
    )

    assert resolved == set()
    assert unresolved == {
        "CG99",
    }


def test_only_selected_candidate_references_are_collected() -> None:
    synthesis = {
        "established_findings": [
            {
                "claim_id": "EF1",
                "evidence_record_ids": [
                    "https://openalex.org/W100",
                ],
            },
            {
                "claim_id": "EF2",
                "evidence_record_ids": [
                    "https://openalex.org/W200",
                ],
            },
        ],
        "unresolved_questions": [],
        "candidate_gaps": [],
    }

    candidates = {
        "candidates": [
            {
                "candidate_id": "selected",
                "novelty_evidence_ids": [
                    "EF1",
                ],
                "feasibility_evidence_ids": [],
            },
            {
                "candidate_id": "rejected",
                "novelty_evidence_ids": [
                    "https://openalex.org/W999",
                ],
                "feasibility_evidence_ids": [],
            },
        ]
    }

    decision = {
        "selected_candidate_id": "selected",
        "evidence_record_ids": [
            "EF1",
        ],
    }

    references = collect_referenced_ids(
        synthesis=synthesis,
        candidates=candidates,
        decision=decision,
    )

    assert (
        "https://openalex.org/W999"
        not in references
    )

    assert (
        "https://openalex.org/W100"
        in references
    )    


def test_malformed_openalex_techrxiv_id_normalises_to_doi() -> None:
    malformed = (
        "https://openalex.org/"
        "W36227/"
        "techrxiv.173386065.57486944/v1"
    )

    assert normalise_evidence_id(
        malformed
    ) == (
        "10.36227/"
        "techrxiv.173386065.57486944/v1"
    )


def test_malformed_openalex_techrxiv_id_resolves_against_doi_record() -> None:
    records = [
        {
            "record_id": (
                "doi:10.36227/"
                "techrxiv.173386065.57486944/v1"
            ),
            "doi": (
                "10.36227/"
                "techrxiv.173386065.57486944/v1"
            ),
            "title": (
                "A Survey on Large Language Models "
                "for Network Operations and Management"
            ),
            "publication_year": 2024,
            "url": (
                "https://doi.org/"
                "10.36227/"
                "techrxiv.173386065.57486944/v1"
            ),
            "abstract": "Example abstract.",
        }
    ]

    alias_index = build_evidence_alias_index(
        records
    )

    malformed = (
        "https://openalex.org/"
        "W36227/"
        "techrxiv.173386065.57486944/v1"
    )

    assert normalise_evidence_id(
        malformed
    ) in alias_index    
    

def test_uniquely_truncated_openalex_id_is_recovered() -> None:
    records = [
        {
            "record_id": (
                "https://openalex.org/W4412158322"
            ),
            "openalex_id": (
                "https://openalex.org/W4412158322"
            ),
            "title": "Example work",
            "publication_year": 2026,
            "abstract": "Example abstract.",
        }
    ]

    alias_index = build_evidence_alias_index(
        records
    )

    recovered = (
        recover_uniquely_truncated_openalex_id(
            "https://openalex.org/W441215832",
            alias_index=alias_index,
        )
    )

    assert recovered == (
        "https://openalex.org/w4412158322"
    )
    
def test_ambiguous_truncated_openalex_id_is_not_recovered() -> None:
    records = [
        {
            "record_id": (
                "https://openalex.org/W4412158321"
            ),
        },
        {
            "record_id": (
                "https://openalex.org/W4412158322"
            ),
        },
    ]

    alias_index = build_evidence_alias_index(
        records
    )

    recovered = (
        recover_uniquely_truncated_openalex_id(
            "https://openalex.org/W441215832",
            alias_index=alias_index,
        )
    )

    assert recovered is None
    
def test_verification_recovers_uniquely_truncated_openalex_id() -> None:
    records = [
        {
            "record_id": (
                "https://openalex.org/W4412158322"
            ),
            "openalex_id": (
                "https://openalex.org/W4412158322"
            ),
            "url": (
                "https://openalex.org/W4412158322"
            ),
            "title": "Example work",
            "publication_year": 2026,
            "abstract": "Example abstract.",
        }
    ]

    synthesis = {
        "established_findings": [
            {
                "claim_id": "EF1",
                "claim": "Example finding.",
                "evidence_record_ids": [
                    "https://openalex.org/W441215832"
                ],
            }
        ],
        "unresolved_questions": [],
        "candidate_gaps": [],
    }

    candidates = {
        "candidates": []
    }

    decision = {
        "selected_candidate_id": None,
        "evidence_record_ids": [],
    }

    report = verify_evidence(
        records=records,
        synthesis=synthesis,
        candidates=candidates,
        decision=decision,
    )

    assert report.missing_record_ids == []
    assert report.critical_issues == []
    assert any(
        "uniquely truncated openalex"
        in warning.lower()
        for warning in report.warnings
    )    
