from cnsm_agentic.autonomous_research.publication_renderer import (
    _render_references,
)


def test_reference_renderer_strips_provider_numeric_prefix():
    records = [
        {
            "record_id": "rec-1",
            "authors": ["[1] R. Example"],
            "title": "Validator Study",
            "venue": "Example Journal",
            "year": "2026",
            "doi": "10.1000/example",
        }
    ]

    rendered = _render_references(
        cited_record_ids=["rec-1"],
        verified_records=records,
    )

    assert rendered.startswith(r"\bibitem{ref1} ")
    assert "[1]" not in rendered
    assert "R. Example" in rendered


def test_reference_renderer_preserves_normal_metadata():
    records = [
        {
            "record_id": "rec-1",
            "authors": ["R. Example"],
            "title": "Validator Study",
            "venue": "Example Journal",
            "year": "2026",
        }
    ]

    rendered = _render_references(
        cited_record_ids=["rec-1"],
        verified_records=records,
    )

    assert r"\bibitem{ref1}" in rendered
    assert "R. Example" in rendered
