from pathlib import Path

from cnsm_agentic.autonomous_research.publication_renderer import (
    build_publication_artifacts,
    render_ieee_latex,
)

from types import SimpleNamespace

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


def test_build_publication_artifacts_compiles_exactly_five_pages(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "cnsm_agentic.autonomous_research."
        "publication_renderer._pdf_page_count",
        lambda _: 5,
    )

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
    assert validation["page_count"] == 5
    assert validation["within_page_limit"] is True
    assert validation["uses_full_page_budget"] is True
    assert validation["disclosure_included"] is True
    assert validation["references_included"] is True
    assert validation["passed"] is True

    assert (tmp_path / "manuscript.tex").is_file()
    assert (tmp_path / "manuscript.pdf").is_file()
    assert (tmp_path / "compilation.log").is_file()

    assert validation["tex_sha256"]
    assert validation["pdf_sha256"]


def test_publication_underfilled_four_pages_fails(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "cnsm_agentic.autonomous_research."
        "publication_renderer._pdf_page_count",
        lambda _: 4,
    )

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
    assert validation["page_count"] == 4
    assert validation["within_page_limit"] is True
    assert validation["uses_full_page_budget"] is False
    assert validation["references_included"] is True
    assert validation["disclosure_included"] is True
    assert validation["passed"] is False


def test_publication_over_limit_six_pages_fails(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "cnsm_agentic.autonomous_research."
        "publication_renderer._pdf_page_count",
        lambda _: 6,
    )

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
    assert validation["page_count"] == 6
    assert validation["within_page_limit"] is False
    assert validation["uses_full_page_budget"] is False
    assert validation["references_included"] is True
    assert validation["disclosure_included"] is True
    assert validation["passed"] is False


def test_render_ieee_latex_normalizes_problematic_unicode():
    manuscript = _manuscript()
    manuscript["abstract"] = (
        "Guarded − baseline; "
        "range 1–2; test—case; "
        "target ≥ 30%; "
        "value ≤ 5; "
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
    assert "≥" not in source
    assert "≤" not in source

    assert "Guarded - baseline" in source
    assert "range 1--2" in source
    assert "test---case" in source
    assert r"target >= 30\%" in source
    assert "value <= 5" in source

    assert "Kovačić" in source
    assert r"\usepackage[utf8]{inputenc}" in source
    assert r"\usepackage[T1]{fontenc}" in source


def test_publication_renderer_uses_robust_subprocess_decoding(
    tmp_path: Path,
    monkeypatch,
):
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs)

        pdf_path = tmp_path / "manuscript.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        return SimpleNamespace(
            returncode=0,
            stdout="latex output \ufffd",
            stderr="",
        )

    monkeypatch.setattr(
        "cnsm_agentic.autonomous_research."
        "publication_renderer.subprocess.run",
        fake_run,
    )

    monkeypatch.setattr(
        "cnsm_agentic.autonomous_research."
        "publication_renderer._pdf_page_count",
        lambda _: 5,
    )

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

    assert observed["text"] is True
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
    assert validation["page_count"] == 5
