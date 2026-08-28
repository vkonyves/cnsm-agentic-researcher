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
    assert (
        "Do not merely delete offending metadata."
        in PIPELINE
    )

    assert (
        "replace it with concise prose grounded in the supplied verified"
        in PIPELINE
    )

    assert (
        "literature, execution evidence, or completed analysis"
        in PIPELINE
    )

    assert (
        "the purpose of each section"
        in PIPELINE
    )

    assert (
        "preserve scientific information"
        in PIPELINE
    )

    assert (
        "generic filler"
        in PIPELINE
    )


def test_positive_section_specific_scientific_prose_is_required():
    assert (
        "Related-work prose should summarize and compare prior"
        in PIPELINE
    )

    assert (
        "methodology prose should explain"
        in PIPELINE
    )

    assert (
        "results prose should report and"
        in PIPELINE
    )

    assert (
        "discussion prose should explain"
        in PIPELINE
    )
