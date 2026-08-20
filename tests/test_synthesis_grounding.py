from cnsm_agentic.autonomous_research.pipeline import (
    _synthesis_grounding_issues,
)
from cnsm_agentic.autonomous_research.schemas import (
    EvidenceClaim,
    EvidenceSynthesis,
)


def _claim(
    evidence_ids: list[str],
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="EF1",
        statement="Supported claim.",
        evidence_record_ids=evidence_ids,
        evidence_type="empirical",
        confidence=0.9,
        limitations=[],
    )


def test_synthesis_grounding_accepts_supplied_record_id():
    synthesis = EvidenceSynthesis(
        established_findings=[
            _claim(["record-1"])
        ],
        unresolved_questions=[],
        candidate_gaps=[],
    )

    issues = _synthesis_grounding_issues(
        synthesis,
        [
            {
                "record_id": "record-1",
                "doi": None,
            }
        ],
    )

    assert issues == []


def test_synthesis_grounding_accepts_equivalent_doi():
    synthesis = EvidenceSynthesis(
        established_findings=[
            _claim(["doi:10.1234/example"])
        ],
        unresolved_questions=[],
        candidate_gaps=[],
    )

    issues = _synthesis_grounding_issues(
        synthesis,
        [
            {
                "record_id": "record-1",
                "doi": (
                    "https://doi.org/"
                    "10.1234/example"
                ),
            }
        ],
    )

    assert issues == []


def test_synthesis_grounding_rejects_unsupplied_id():
    synthesis = EvidenceSynthesis(
        established_findings=[
            _claim([
                "https://openalex.org/W4413016958"
            ])
        ],
        unresolved_questions=[],
        candidate_gaps=[],
    )

    issues = _synthesis_grounding_issues(
        synthesis,
        [
            {
                "record_id": (
                    "https://openalex.org/W123"
                ),
                "doi": "10.1234/example",
            }
        ],
    )

    assert len(issues) == 1
    assert "W4413016958" in issues[0]
    assert "was not supplied" in issues[0]


def test_synthesis_grounding_does_not_accept_unsupplied_alias():
    synthesis = EvidenceSynthesis(
        established_findings=[
            _claim([
                "https://openalex.org/W999"
            ])
        ],
        unresolved_questions=[],
        candidate_gaps=[],
    )

    issues = _synthesis_grounding_issues(
        synthesis,
        [
            {
                "record_id": "crossref:abc",
                "doi": "10.1234/example",
            }
        ],
    )

    assert issues
