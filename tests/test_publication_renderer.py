from pathlib import Path

from cnsm_agentic.autonomous_research.publication_renderer import (
    build_publication_artifacts,
    render_ieee_latex,
)


def _manuscript():
    return {
        "title": "Test Autonomous NetOps Study",
        "abstract": "A short abstract.",
        "sections": {
            "introduction": "Introduction text.",
            "related_work": "Related work text.",
            "methodology": "Methods text.",
            "results": "Results text.",
            "discussion": "Discussion text.",
            "conclusion": "Conclusion text.",
        },
        "figure_captions": [],
        "table_captions": [],
        "cited_record_ids": ["record-1"],
        "disclosure_statement": (
            "This manuscript was generated autonomously."
        ),
        "limitations": [
            "Synthetic evaluation."
        ],
    }


def _constraints():
    return {
        "format": {
            "latex_document_class": (
                r"\documentclass[conference]{IEEEtran}"
            ),
            "maximum_pages": 5,
            "references_included_in_limit": True,
            "disclosure_statement_mandatory": True,
            "disclosure_statement_included_in_limit": True,
            "disclosure_statement_must_be_within_pages_1_to_5": True,
            "sixth_page_disclosure_prohibited": True,
            "template_manipulation_prohibited": True,
        }
    }


def test_render_ieee_latex_contains_required_components():
    source = render_ieee_latex(
        manuscript=_manuscript(),
        verified_records=[
            {
                "record_id": "record-1",
                "title": "Verified Paper",
                "publication_year": 2026,
            }
        ],
        document_class=(
            r"\documentclass[conference]{IEEEtran}"
        ),
    )

    assert (
        r"\documentclass[conference]{IEEEtran}"
        in source
    )
    assert r"\section*{Disclosure Statement}" in source
    assert r"\begin{thebibliography}{99}" in source
    assert "Verified Paper" in source


def test_build_publication_artifacts_compiles_pdf(
    tmp_path: Path,
):
    validation = build_publication_artifacts(
        manuscript=_manuscript(),
        verified_records=[
            {
                "record_id": "record-1",
                "title": "Verified Paper",
                "publication_year": 2026,
            }
        ],
        output_dir=tmp_path,
        paper_run_constraints=_constraints(),
    )

    assert validation["compile_status"] == "passed"
    assert validation["page_count"] is not None
    assert validation["page_count"] <= 5
    assert validation["within_page_limit"] is True
    assert validation["disclosure_included"] is True
    assert validation["references_included"] is True
    assert validation["passed"] is True

    assert (tmp_path / "manuscript.tex").is_file()
    assert (tmp_path / "manuscript.pdf").is_file()
    assert (tmp_path / "compilation.log").is_file()

    assert validation["tex_sha256"]
    assert validation["pdf_sha256"]

def test_render_ieee_latex_normalizes_problematic_unicode():
    manuscript = _manuscript()
    manuscript["abstract"] = (
        "Guarded − baseline; "
        "range 1–2; test—case; "
        "Kovačić."
    )

    source = render_ieee_latex(
        manuscript=manuscript,
        verified_records=[
            {
                "record_id": "record-1",
                "title": "Verified Paper",
                "publication_year": 2026,
            }
        ],
        document_class=r"\documentclass[conference]{IEEEtran}",
    )

    assert "−" not in source
    assert "–" not in source
    assert "—" not in source
    assert "Guarded - baseline" in source
    assert "range 1--2" in source
    assert "test---case" in source
    assert "Kovačić" in source
    assert r"\usepackage[utf8]{inputenc}" in source
    assert r"\usepackage[T1]{fontenc}" in source
