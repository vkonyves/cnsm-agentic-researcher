from pydantic import BaseModel

from cnsm_agentic.autonomous_research.final_pipeline import (
    sanitize_structured_manuscript_publication_metadata,
)


class DummyManuscript(BaseModel):
    body: str


def _sanitize_text(value: str) -> str:
    result = sanitize_structured_manuscript_publication_metadata(
        DummyManuscript(body=value)
    )
    return result.body


def test_standalone_references_heading_is_removed():
    assert _sanitize_text("References") == ""


def test_markdown_references_heading_is_removed():
    assert _sanitize_text("## References") == ""


def test_scientific_use_of_references_word_is_preserved():
    text = "Prior references establish the verification context."
    assert _sanitize_text(text) == text


def test_reviewer_request_sentence_is_removed_but_science_preserved():
    text = (
        "The guarded arm improved correctness. "
        "Reviewers requested additional discussion of limitations. "
        "The confidence interval was computed by paired bootstrap."
    )

    cleaned = _sanitize_text(text)

    assert "Reviewers requested" not in cleaned
    assert "The guarded arm improved correctness." in cleaned
    assert "The confidence interval was computed by paired bootstrap." in cleaned


def test_publication_sanitizer_removes_embedded_references_heading():
    manuscript = DummyManuscript(
        body="""The experiment supports the bounded conclusion.

References
This sentence discusses references in ordinary prose and must remain."""
    )

    sanitized = sanitize_structured_manuscript_publication_metadata(
        manuscript
    )

    assert "\nReferences\n" not in sanitized.body
    assert "discusses references in ordinary prose" in sanitized.body


def test_publication_sanitizer_removes_latex_references_heading():
    manuscript = DummyManuscript(
        body="""Supported scientific discussion.
\\section*{References}
Further supported scientific discussion."""
    )

    sanitized = sanitize_structured_manuscript_publication_metadata(
        manuscript
    )

    assert "\\section*{References}" not in sanitized.body
    assert "Supported scientific discussion." in sanitized.body
    assert "Further supported scientific discussion." in sanitized.body


def test_sanitizer_removes_numbered_reference_block_but_preserves_inline_citation():
    manuscript = {
        "abstract": "Abstract.",
        "sections": {
            "introduction": (
                "Prior work [1] motivates the experiment.\n\n"
                "[1] A. Author, First duplicate reference.\n"
                "[2] B. Author, Second duplicate reference.\n"
                "[3] C. Author, Third duplicate reference.\n\n"
                "Scientific discussion continues here."
            ),
            "related_work": "Related work.",
            "methodology": "Methodology.",
            "results": "Results.",
            "discussion": "Discussion.",
            "limitations": ["Limitation."],
            "conclusion": "Conclusion.",
        },
        "references": [],
        "disclosure_statement": "Disclosure.",
    }

    sanitized = sanitize_structured_manuscript_publication_metadata(
        manuscript
    )

    introduction = sanitized["sections"]["introduction"]

    assert "Prior work [1] motivates the experiment." in introduction
    assert "Scientific discussion continues here." in introduction

    assert "[1] A. Author" not in introduction
    assert "[2] B. Author" not in introduction
    assert "[3] C. Author" not in introduction
