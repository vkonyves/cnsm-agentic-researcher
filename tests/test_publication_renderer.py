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
        "baseline ≈ 40%; "
        "difference ≠ 0; "
        "uncertainty ± 2%; "
        "A → B; "
        "rate ∝ load; "
        "limit → ∞; "
        "temperature 25°C; "
        "parameter μ = 0.5; "
        "micro sign µ = 0.4; "
        "matrix Λ = 1; "
        "parameters Γ, Δ, Θ, Ξ, Π, Σ, Υ, Φ, Ψ, Ω; "
        "lowercase α, β, γ, δ, ε, ζ, η, θ, ι, κ, "
        "λ, ν, ξ, π, ρ, σ, τ, υ, φ, χ, ψ, ω; "
        "relation x ∼ y; "
        "approximation x ≃ y; "
        "identity a ≡ b; "
        "product x ⋅ y; "
        "root √x; "
        "set x ∈ S; "
        "CO₂ measurement; "
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

    # Unicode punctuation/symbols should not survive into raw LaTeX.
    assert "−" not in source
    assert "–" not in source
    assert "—" not in source
    assert "≥" not in source
    assert "≤" not in source
    assert "≈" not in source
    assert "≠" not in source
    assert "±" not in source
    assert "→" not in source
    assert "∝" not in source
    assert "∞" not in source
    assert "°" not in source
    assert "μ" not in source
    assert "µ" not in source
    assert "Λ" not in source
    assert "Γ" not in source
    assert "Δ" not in source
    assert "Θ" not in source
    assert "Ξ" not in source
    assert "Π" not in source
    assert "Σ" not in source
    assert "Υ" not in source
    assert "Φ" not in source
    assert "Ψ" not in source
    assert "Ω" not in source
    assert "α" not in source
    assert "β" not in source
    assert "γ" not in source
    assert "δ" not in source
    assert "ε" not in source
    assert "ζ" not in source
    assert "η" not in source
    assert "θ" not in source
    assert "ι" not in source
    assert "κ" not in source
    assert "λ" not in source
    assert "ν" not in source
    assert "ξ" not in source
    assert "π" not in source
    assert "ρ" not in source
    assert "σ" not in source
    assert "τ" not in source
    assert "υ" not in source
    assert "φ" not in source
    assert "χ" not in source
    assert "ψ" not in source
    assert "ω" not in source
    assert "∼" not in source
    assert "≃" not in source
    assert "≡" not in source
    assert "⋅" not in source
    assert "√" not in source
    assert "∈" not in source
    assert "₂" not in source

    # Text-normalized punctuation.
    assert "Guarded - baseline" in source
    assert "range 1--2" in source
    assert "test---case" in source
    assert r"target >= 30\%" in source
    assert "value <= 5" in source

    # Mathematical/scientific symbol replacements.
    assert r"\ensuremath{\approx}" in source
    assert r"\ensuremath{\neq}" in source
    assert r"\ensuremath{\pm}" in source
    assert r"\ensuremath{\rightarrow}" in source
    assert r"\ensuremath{\propto}" in source
    assert r"\ensuremath{\infty}" in source
    assert r"\ensuremath{^\circ}" in source
    assert r"\ensuremath{\mu}" in source
    assert r"\ensuremath{\Lambda}" in source
    assert r"\ensuremath{\Gamma}" in source
    assert r"\ensuremath{\Delta}" in source
    assert r"\ensuremath{\Theta}" in source
    assert r"\ensuremath{\Xi}" in source
    assert r"\ensuremath{\Pi}" in source
    assert r"\ensuremath{\Sigma}" in source
    assert r"\ensuremath{\Upsilon}" in source
    assert r"\ensuremath{\Phi}" in source
    assert r"\ensuremath{\Psi}" in source
    assert r"\ensuremath{\Omega}" in source

    assert r"\ensuremath{\alpha}" in source
    assert r"\ensuremath{\beta}" in source
    assert r"\ensuremath{\gamma}" in source
    assert r"\ensuremath{\delta}" in source
    assert r"\ensuremath{\epsilon}" in source
    assert r"\ensuremath{\zeta}" in source
    assert r"\ensuremath{\eta}" in source
    assert r"\ensuremath{\theta}" in source
    assert r"\ensuremath{\iota}" in source
    assert r"\ensuremath{\kappa}" in source
    assert r"\ensuremath{\lambda}" in source
    assert r"\ensuremath{\nu}" in source
    assert r"\ensuremath{\xi}" in source
    assert r"\ensuremath{\pi}" in source
    assert r"\ensuremath{\rho}" in source
    assert r"\ensuremath{\sigma}" in source
    assert r"\ensuremath{\tau}" in source
    assert r"\ensuremath{\upsilon}" in source
    assert r"\ensuremath{\phi}" in source
    assert r"\ensuremath{\chi}" in source
    assert r"\ensuremath{\psi}" in source
    assert r"\ensuremath{\omega}" in source

    assert r"\ensuremath{\sim}" in source
    assert r"\ensuremath{\simeq}" in source
    assert r"\ensuremath{\equiv}" in source
    assert r"\ensuremath{\cdot}" in source
    assert r"\ensuremath{\sqrt{}}" in source
    assert r"\ensuremath{\in}" in source
    assert r"\textsubscript{2}" in source

    # Unicode author names remain supported through UTF-8/T1.
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
