from pathlib import Path

from cnsm_agentic.autonomous_research.final_pipeline import (
    audit_manuscript_publication_sanity,
)


def _write_master_prompt_sha(
    root: Path,
    digest: str = (
        "1872df1e1805d2d96940456ca016bd665"
        "d1d5196add77f5acdf1582bb39b15ba"
    ),
) -> None:
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    (provenance / "master_prompt.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )


def _write_final(
    root: Path,
    *,
    tex: str,
    log: str,
) -> None:
    _write_master_prompt_sha(root)

    final = root / "manuscript" / "final"
    final.mkdir(parents=True)

    (final / "manuscript.tex").write_text(
        tex,
        encoding="utf-8",
    )

    (final / "manuscript.log").write_text(
        log,
        encoding="utf-8",
    )

def test_clean_publication_sanity_passes(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is True



def test_missing_master_prompt_hash_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Autonomous disclosure without the required digest.

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["full_sha256_count"] == 0


def test_wrong_single_master_prompt_hash_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Master prompt SHA-256 =
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["full_sha256_count"] == 1
    assert any(
        "does not match" in issue
        for issue in audit["issues"]
    )


def test_renderer_allowbreak_master_hash_passes(tmp_path):
    digest = (
        "1872df1e1805d2d96940456ca016bd665"
        "d1d5196add77f5acdf1582bb39b15ba"
    )
    rendered_digest = r"\allowbreak{}".join(
        digest[i:i + 8]
        for i in range(0, 64, 8)
    )

    _write_final(
        tmp_path,
        tex=(
            "Scientific prose.\n"
            "\\section*{Disclosure Statement}\n"
            "Master prompt SHA-256 = "
            + rendered_digest
            + "\n"
            "\\begin{thebibliography}{1}\n"
            "\\bibitem{x} A verified paper.\n"
            "\\end{thebibliography}\n"
        ),
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is True
    assert audit["metrics"]["full_sha256_count"] == 1


def test_second_references_heading_before_bibliography_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{References}
[1] Manually generated reference text.

\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert (
        audit["metrics"][
            "prebibliography_references_heading_count"
        ]
        == 1
    )


def test_multiple_bibliography_environments_fail(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{1}
\bibitem{x} First paper.
\end{thebibliography}

\begin{thebibliography}{1}
\bibitem{y} Second paper.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["bibliography_environment_count"] == 2


def test_duplicate_bibliography_doi_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{2}
\bibitem{a} First citation, 10.18653/v1/2023.emnlp-main.187.
\bibitem{b} Duplicate citation, 10.18653/V1/2023.EMNLP-MAIN.187.
\end{thebibliography}
""",
        log="Output written on manuscript.pdf (5 pages).\n",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert (
        audit["metrics"]["duplicate_bibliography_doi_count"]
        == 1
    )


def test_multiple_full_hashes_fail(tmp_path):
    _write_final(
        tmp_path,
        tex=(
            "a" * 64
            + "\n"
            + "b" * 64
        ),
        log="",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["full_sha256_count"] == 2


def test_large_overfull_hbox_fails(tmp_path):
    _write_final(
        tmp_path,
        tex="Scientific prose.",
        log=(
            "Overfull \\hbox "
            "(56.7887pt too wide) in paragraph\n"
        ),
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert (
        audit["metrics"]["significant_overfull_count"]
        == 1
    )


def test_small_overfull_hbox_is_tolerated(tmp_path):
    _write_final(
        tmp_path,
        tex=r"""
Scientific prose.

\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
""",
        log=(
            "Overfull \\hbox "
            "(1.5pt too wide) in paragraph\n"
        ),
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is True


def test_reviewer_meta_language_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=(
            "If reviewers require these values, "
            "we will insert them."
        ),
        log="",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert (
        audit["metrics"]["reviewer_meta_phrase_count"]
        >= 1
    )


def test_inline_doi_label_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=(
            "Prior work [1] (DOI:10.1000/example).\n"
            r"\begin{thebibliography}{1}"
            "\nReference\n"
        ),
        log="",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["inline_doi_label_count"] == 1


def test_raw_reproduction_command_fails(tmp_path):
    _write_final(
        tmp_path,
        tex=(
            "Reproduce with python3 execution/tools/run.py "
            "--input analysis/results.json\n"
        ),
        log="",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert audit["metrics"]["raw_command_count"] >= 1


def test_excessive_artifact_paths_are_diagnostic_only(tmp_path):
    paths = " ".join(
        f"analysis/file_{i}.json"
        for i in range(9)
    )

    tex = (
        paths
        + "\n\n"
        + r"""
\section*{Disclosure Statement}
Master prompt SHA-256 =
1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba

\begin{thebibliography}{1}
\bibitem{x} A verified paper.
\end{thebibliography}
"""
    )

    _write_final(
        tmp_path,
        tex=tex,
        log="",
    )

    audit = audit_manuscript_publication_sanity(
        run_dir=tmp_path,
    )

    assert audit["passed"] is True
    assert audit["metrics"]["artifact_path_count"] == 9
    assert audit["metrics"]["artifact_path_count"] == 9


def test_undefined_citation_warning_fails_publication_sanity(
    tmp_path,
):
    final_dir = tmp_path / "manuscript" / "final"
    final_dir.mkdir(parents=True)

    (final_dir / "manuscript.tex").write_text(
        r"\begin{document}Scientific text.\end{document}",
        encoding="utf-8",
    )

    (final_dir / "manuscript.log").write_text(
        "LaTeX Warning: Citation `ref1' on page 2 undefined "
        "on input line 42.\n"
        "There were undefined references.\n",
        encoding="utf-8",
    )

    result = audit_manuscript_publication_sanity(
        run_dir=tmp_path
    )

    assert result["passed"] is False
    assert (
        result["metrics"][
            "undefined_reference_warning_count"
        ]
        >= 1
    )


def test_missing_character_warning_fails_publication_sanity(
    tmp_path,
):
    final_dir = tmp_path / "manuscript" / "final"
    final_dir.mkdir(parents=True)

    (final_dir / "manuscript.tex").write_text(
        r"\begin{document}Scientific text.\end{document}",
        encoding="utf-8",
    )

    (final_dir / "manuscript.log").write_text(
        "Missing character: There is no Ω in font ptmr7t!\n",
        encoding="utf-8",
    )

    result = audit_manuscript_publication_sanity(
        run_dir=tmp_path
    )

    assert result["passed"] is False
    assert (
        result["metrics"][
            "missing_character_warning_count"
        ]
        == 1
    )


def _write_master_prompt_sha(
    run_dir,
    digest=(
        "1872df1e1805d2d96940456ca016bd665"
        "d1d5196add77f5acdf1582bb39b15ba"
    ),
):
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (
        provenance_dir / "master_prompt.sha256"
    ).write_text(
        digest + "\n",
        encoding="utf-8",
    )


def _write_execution_manifest_for_publication_sanity(root: Path) -> None:
    execution = root / "execution"
    execution.mkdir(parents=True, exist_ok=True)
    (execution / "execution_manifest.json").write_text(
        '{"controlled_fault_assignment_sha256":"8d0879872348ae94a45f3d674df57d6552624cd841b62ab239a280347f5275c6","source_generation_attempt_count":40,"planned_episode_count":80}\n',
        encoding="utf-8",
    )


def _write_analysis_results_for_publication_sanity(root: Path) -> None:
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "results.json").write_text(
        '{"secondary_results":[]}\n',
        encoding="utf-8",
    )


def test_prebibliography_numbered_reference_list_fails(tmp_path):
    tex = (
        "Scientific prose.\n"
        "[1] A. Author, First paper.\n"
        "[2] B. Author, Second paper.\n"
        "\\section*{Disclosure Statement}\n"
        "Master prompt SHA-256 =\n"
        "1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba\n"
        "\\begin{thebibliography}{2}\n"
        "\\bibitem{x} A paper.\n"
        "\\bibitem{y} Another paper.\n"
        "\\end{thebibliography}\n"
    )
    _write_final(tmp_path, tex=tex, log="")
    audit = audit_manuscript_publication_sanity(run_dir=tmp_path)
    assert audit["passed"] is False
    assert audit["metrics"]["prebibliography_numbered_reference_count"] == 2


def test_empty_sha256_assignment_fails(tmp_path):
    tex = (
        "Scientific prose.\n"
        'controlled_fault_assignment_sha256 = ""\n'
        "\\section*{Disclosure Statement}\n"
        "Master prompt SHA-256 =\n"
        "1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba\n"
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{x} A paper.\n"
        "\\end{thebibliography}\n"
    )
    _write_final(tmp_path, tex=tex, log="")
    audit = audit_manuscript_publication_sanity(run_dir=tmp_path)
    assert audit["passed"] is False
    assert audit["metrics"]["empty_sha256_assignment_count"] == 1


def test_unexecuted_harmful_overrepair_claim_fails(tmp_path):
    _write_analysis_results_for_publication_sanity(tmp_path)
    tex = (
        "Scientific prose.\n"
        "No preregistered harmful-overrepair events were observed.\n"
        "\\section*{Disclosure Statement}\n"
        "Master prompt SHA-256 =\n"
        "1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba\n"
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{x} A paper.\n"
        "\\end{thebibliography}\n"
    )
    _write_final(tmp_path, tex=tex, log="")
    audit = audit_manuscript_publication_sanity(run_dir=tmp_path)
    assert audit["passed"] is False
    assert audit["metrics"]["unsupported_harmful_overrepair_claim_count"] >= 1


def test_unsupported_source_retry_claim_fails(tmp_path):
    _write_execution_manifest_for_publication_sanity(tmp_path)
    tex = (
        "Scientific prose.\n"
        "One source candidate failed validation after the single automatic retry.\n"
        "\\section*{Disclosure Statement}\n"
        "Master prompt SHA-256 =\n"
        "1872df1e1805d2d96940456ca016bd665d1d5196add77f5acdf1582bb39b15ba\n"
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{x} A paper.\n"
        "\\end{thebibliography}\n"
    )
    _write_final(tmp_path, tex=tex, log="")
    audit = audit_manuscript_publication_sanity(run_dir=tmp_path)
    assert audit["passed"] is False
    assert audit["metrics"]["unsupported_source_retry_claim_count"] >= 1

