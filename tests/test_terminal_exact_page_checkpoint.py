from pathlib import Path


def source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")


def test_checkpoint_is_created_before_terminal_review():
    text = source()

    checkpoint = text.index(
        "protected_submission_manuscript:"
    )
    terminal_review = text.index(
        "Terminal AI peer review "
    )

    assert checkpoint < terminal_review
    assert (
        'publication_validation.get("passed") is True'
        in text
    )


def test_terminal_review_starts_from_checkpoint():
    text = source()

    assert (
        "terminal_review_revision_base_manuscript = ("
        in text
    )
    assert (
        "protected_submission_manuscript"
        in text
    )
    assert (
        "Make only the minimum scientific edits"
        in text
    )
    assert "Do not globally" in text
    assert "rewrite" in text
    assert "summarize" in text
    assert "condense" in text


def test_underfilled_terminal_revision_restarts_from_checkpoint():
    text = source()

    assert (
        "terminal_page_count < terminal_maximum_pages"
        in text
    )
    assert (
        "and protected_submission_manuscript is not None"
        in text
    )
    assert (
        "terminal_revision_base_manuscript = ("
        in text
    )


def test_overfull_candidate_remains_compaction_base():
    text = source()

    marker = (
        "if terminal_page_count > terminal_maximum_pages:"
    )
    start = text.index(marker)
    section = text[start:start + 900]

    assert "revised_manuscript" in section
    assert (
        "terminal_revision_base_publication_validation"
        in section
    )


def test_exact_terminal_candidate_can_promote_checkpoint():
    text = source()

    assert (
        "selected_exact_terminal_candidate = ("
        in text
    )
    assert (
        '"page_count"'
        in text
    )
    assert (
        '"maximum_pages"'
        in text
    )


def test_short_terminal_candidate_cannot_replace_checkpoint():
    text = source()

    marker = (
        "The terminal rewrite/convergence did not preserve"
    )

    start = text.index(marker)
    section = text[start:start + 1000]

    assert "protected_submission_manuscript" in section
    assert "protected_submission_validation" in section


def test_authoritative_pdf_is_rerendered_after_selection():
    text = source()

    marker = (
        "Re-render exactly the manuscript that is actually carried"
    )

    start = text.index(marker)
    section = text[start:start + 1300]

    assert "build_publication_artifacts(" in section