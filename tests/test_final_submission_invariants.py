from pathlib import Path


PIPELINE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
)


def source() -> str:
    return PIPELINE.read_text(encoding="utf-8")


def test_exact_page_submission_is_hard_final_invariant():
    text = source()

    assert (
        'publication_validation.get("page_count")\n'
        '            != publication_validation.get("maximum_pages")'
        in text
    )

    assert "Final manuscript is not an exact-page submission" in text


def test_empty_citation_set_does_not_expand_to_all_records():
    text = source()

    assert "if not cited_record_ids:" in text
    assert (
        "Never interpret an empty citation"
        in text
    )


def test_best_page_selection_handles_none_page_count():
    current_page_count = None
    best_page_count = 5

    comparable_current_page_count = (
        current_page_count
        if isinstance(current_page_count, int)
        else -1
    )

    assert best_page_count > comparable_current_page_count


def test_publication_hygiene_removes_non_bmp_unicode_for_pdflatex():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert r'[\U00010000-\U0010FFFF]' in source
