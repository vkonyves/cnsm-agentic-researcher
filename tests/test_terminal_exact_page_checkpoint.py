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


def test_terminal_review_remediation_is_cumulative():
    text = source()

    marker = (
        "terminal_review_revision_base_manuscript = ("
    )
    start = text.index(marker)
    section = text[start:start + 900]

    # Each terminal revision must start from the manuscript
    # produced by the immediately preceding review/revision
    # round, not from the original protected checkpoint.
    assert (
        "terminal_review_revision_base_manuscript = ("
        in section
    )
    assert "revised_manuscript" in section
    assert (
        "terminal_review_revision_base_validation = ("
        in section
    )
    assert "publication_validation" in section

    # The checkpoint must not be selected as the normal
    # scientific revision base.
    assert (
        "protected_submission_manuscript"
        not in section
    )
    assert (
        "protected_submission_validation"
        not in section
    )

    # Local-edit constraints remain part of the revision prompt.
    assert (
        "Make only the minimum scientific edits"
        in text
    )
    assert "Do not globally" in text
    assert "rewrite" in text
    assert "summarize" in text
    assert "condense" in text


def test_underfilled_terminal_formatting_preserves_reviewed_candidate():
    text = source()

    marker = (
        "if terminal_page_count < terminal_maximum_pages:"
    )
    start = text.index(marker)
    section = text[start:start + 1400]

    assert (
        "terminal_page_count < terminal_maximum_pages"
        in section
    )

    # Page-format recovery must continue from the best
    # review-remediated candidate rather than silently rolling
    # scientific content back to the protected checkpoint.
    assert "terminal_base = (" in section
    assert "best_terminal_manuscript" in section
    assert "revised_manuscript" in section
    assert (
        "protected_submission_manuscript"
        not in section
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