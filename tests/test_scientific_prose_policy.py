from pathlib import Path


AGENTS = Path(
    "src/cnsm_agentic/autonomous_research/final_agents.py"
).read_text(encoding="utf-8")

PIPELINE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
).read_text(encoding="utf-8")


def test_author_and_reviser_require_positive_scientific_prose():
    assert AGENTS.count(
        "Positively prioritize substantive scientific prose."
    ) == 2

    assert AGENTS.count(
        "Related-work text must synthesize the supplied verified literature"
    ) == 2

    assert AGENTS.count(
        "Preserve scientific information density"
    ) == 2


def test_reviewer_checks_scientific_paper_quality():
    assert (
        "Also evaluate whether the manuscript reads as a scientific conference"
        in AGENTS
    )

    assert (
        "artifact report or audit trail"
        in AGENTS
    )

    assert (
        "literature-grounded scholarly prose"
        in AGENTS
    )

    assert (
        "metadata-heavy prose"
        in AGENTS
    )


def test_hygiene_remediation_replaces_metadata_with_science():
    assert "Do not merely delete offending metadata." in PIPELINE
    assert "CONTENT-PRESERVING" in PIPELINE
    assert "not as summarization, compression" in PIPELINE
    assert "approximately the same rendered scientific volume" in PIPELINE
    assert "SAME already-supported scientific substance" in PIPELINE
    assert "Preserve all unaffected manuscript material" in PIPELINE
    assert "Do not trade scientific completeness for metadata " in PIPELINE
    assert "cleanliness." in PIPELINE


def test_positive_section_specific_scientific_prose_is_required():
    assert (
        "Related-work prose should synthesize and compare prior findings"
        in PIPELINE
    )

    assert (
        "methodology prose should explain"
        in PIPELINE
    )

    assert (
        "results prose should report "
        in PIPELINE
    )
    assert (
        "and interpret completed outcomes"
        in PIPELINE
    )

    assert (
        "discussion prose should explain"
        in PIPELINE
    )


def test_hygiene_remediation_is_not_summarization_or_compression():
    assert (
        "CONTENT-PRESERVING"
        in PIPELINE
    )
    assert (
        "not as summarization, compression"
        in PIPELINE
    )
    assert (
        "approximately the same rendered scientific volume"
        in PIPELINE
    )
    assert (
        "hygiene repair must not turn a full scientific "
        in PIPELINE
    )
    assert (
        "paper into an underfilled one by deleting supported exposition"
        in PIPELINE
    )


def test_hygiene_remediation_preserves_supported_exposition():
    assert (
        "Do not collapse several substantive sentences"
        in PIPELINE
    )
    assert (
        "SAME already-supported scientific substance"
        in PIPELINE
    )
    assert (
        "Do not trade scientific completeness for metadata "
        in PIPELINE
    )
    assert (
        "cleanliness."
        in PIPELINE
    )


def test_manuscript_agents_require_conventional_academic_titles():
    from pathlib import Path

    agents = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert (
        agents.count(
            "Use a conventional academic research-paper title"
        )
        == 2
    )
    assert "Do not use arrow chains" in agents
    assert '"A → B → C"' in agents
    assert '"A -> B -> C"' in agents
    assert "slide-style titles" in agents
    assert "question-based title" in agents
