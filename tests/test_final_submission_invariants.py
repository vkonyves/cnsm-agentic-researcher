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
