from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def _latex_escape(value: str) -> str:
    # Unicode punctuation/spacing that can be safely normalized
    # to plain-text equivalents before LaTeX escaping.
    unicode_text_replacements = {
        "\u2013": "--",     # en dash
        "\u2014": "---",    # em dash
        "\u2212": "-",      # mathematical minus sign
        "\u00a0": " ",      # non-breaking space
        "\u202f": " ",      # narrow no-break space
        "\u2009": " ",      # thin space
        "\u200b": "",       # zero-width space
        "\u2060": "",       # word joiner
        "\u2018": "'",      # left single quote
        "\u2019": "'",      # right single quote
        "\u201c": '"',      # left double quote
        "\u201d": '"',      # right double quote
        "\u2026": "...",    # ellipsis
        "\u2032": "'",      # prime
        "\u2033": "''",     # double prime
        "\u2265": ">=",     # greater than or equal
        "\u2264": "<=",     # less than or equal
    }

    # Scientific symbols that should retain their mathematical
    # meaning in the generated IEEE LaTeX.
    unicode_latex_replacements = {
        "\u2248": r"\ensuremath{\approx}",          # ≈
        "\u2260": r"\ensuremath{\neq}",             # ≠
        "\u00b1": r"\ensuremath{\pm}",              # ±
        "\u2213": r"\ensuremath{\mp}",              # ∓
        "\u00d7": r"\ensuremath{\times}",           # ×
        "\u00f7": r"\ensuremath{\div}",             # ÷
        "\u221e": r"\ensuremath{\infty}",           # ∞
        "\u221d": r"\ensuremath{\propto}",          # ∝
        "\u2202": r"\ensuremath{\partial}",         # ∂
        "\u2207": r"\ensuremath{\nabla}",           # ∇
        "\u2211": r"\ensuremath{\sum}",             # ∑
        "\u220f": r"\ensuremath{\prod}",            # ∏
        "\u222b": r"\ensuremath{\int}",             # ∫
        "\u2208": r"\ensuremath{\in}",              # ∈
        "\u2209": r"\ensuremath{\notin}",           # ∉
        "\u2282": r"\ensuremath{\subset}",          # ⊂
        "\u2286": r"\ensuremath{\subseteq}",        # ⊆
        "\u2283": r"\ensuremath{\supset}",          # ⊃
        "\u2287": r"\ensuremath{\supseteq}",        # ⊇
        "\u222a": r"\ensuremath{\cup}",             # ∪
        "\u2229": r"\ensuremath{\cap}",             # ∩
        "\u2192": r"\ensuremath{\rightarrow}",      # →
        "\u2190": r"\ensuremath{\leftarrow}",       # ←
        "\u2194": r"\ensuremath{\leftrightarrow}",  # ↔
        "\u21d2": r"\ensuremath{\Rightarrow}",      # ⇒
        "\u21d0": r"\ensuremath{\Leftarrow}",       # ⇐
        "\u21d4": r"\ensuremath{\Leftrightarrow}",  # ⇔
        "\u00b0": r"\ensuremath{^\circ}",           # °
        "\u00b7": r"\ensuremath{\cdot}",            # ·
        "\u2022": r"\textbullet{}",                 # •

        # Additional common mathematical/statistical symbols.
        "\u221a": r"\ensuremath{\sqrt{}}",          # √
        "\u223c": r"\ensuremath{\sim}",             # ∼
        "\u2243": r"\ensuremath{\simeq}",           # ≃
        "\u2261": r"\ensuremath{\equiv}",           # ≡
        "\u22c5": r"\ensuremath{\cdot}",            # ⋅

        # Lower-case Greek letters.
        "\u03b1": r"\ensuremath{\alpha}",           # α
        "\u03b2": r"\ensuremath{\beta}",            # β
        "\u03b3": r"\ensuremath{\gamma}",           # γ
        "\u03b4": r"\ensuremath{\delta}",           # δ
        "\u03b5": r"\ensuremath{\epsilon}",         # ε
        "\u03b6": r"\ensuremath{\zeta}",            # ζ
        "\u03b7": r"\ensuremath{\eta}",             # η
        "\u03b8": r"\ensuremath{\theta}",           # θ
        "\u03b9": r"\ensuremath{\iota}",            # ι
        "\u03ba": r"\ensuremath{\kappa}",           # κ
        "\u03bb": r"\ensuremath{\lambda}",          # λ
        "\u03bc": r"\ensuremath{\mu}",              # μ
        "\u00b5": r"\ensuremath{\mu}",              # µ
        "\u03bd": r"\ensuremath{\nu}",              # ν
        "\u03be": r"\ensuremath{\xi}",              # ξ
        "\u03bf": "o",                              # ο
        "\u03c0": r"\ensuremath{\pi}",              # π
        "\u03c1": r"\ensuremath{\rho}",             # ρ
        "\u03c3": r"\ensuremath{\sigma}",           # σ
        "\u03c2": r"\ensuremath{\sigma}",           # ς
        "\u03c4": r"\ensuremath{\tau}",             # τ
        "\u03c5": r"\ensuremath{\upsilon}",         # υ
        "\u03c6": r"\ensuremath{\phi}",             # φ
        "\u03c7": r"\ensuremath{\chi}",             # χ
        "\u03c8": r"\ensuremath{\psi}",             # ψ
        "\u03c9": r"\ensuremath{\omega}",           # ω

        # Upper-case Greek letters with distinct LaTeX commands.
        "\u0393": r"\ensuremath{\Gamma}",           # Γ
        "\u0394": r"\ensuremath{\Delta}",           # Δ
        "\u0398": r"\ensuremath{\Theta}",           # Θ
        "\u039b": r"\ensuremath{\Lambda}",          # Λ
        "\u039e": r"\ensuremath{\Xi}",              # Ξ
        "\u03a0": r"\ensuremath{\Pi}",              # Π
        "\u03a3": r"\ensuremath{\Sigma}",           # Σ
        "\u03a5": r"\ensuremath{\Upsilon}",         # Υ
        "\u03a6": r"\ensuremath{\Phi}",             # Φ
        "\u03a8": r"\ensuremath{\Psi}",             # Ψ
        "\u03a9": r"\ensuremath{\Omega}",           # Ω

        # Superscripts/subscripts commonly emitted in scientific prose.
        "\u00b2": r"\textsuperscript{2}",           # ²
        "\u00b3": r"\textsuperscript{3}",           # ³
        "\u2080": r"\textsubscript{0}",             # ₀
        "\u2081": r"\textsubscript{1}",             # ₁
        "\u2082": r"\textsubscript{2}",             # ₂
        "\u2083": r"\textsubscript{3}",             # ₃
        "\u2084": r"\textsubscript{4}",             # ₄
        "\u2085": r"\textsubscript{5}",             # ₅
        "\u2086": r"\textsubscript{6}",             # ₆
        "\u2087": r"\textsubscript{7}",             # ₇
        "\u2088": r"\textsubscript{8}",             # ₈
        "\u2089": r"\textsubscript{9}",             # ₉
    }

    value = "".join(
        unicode_text_replacements.get(char, char)
        for char in value
    )

    # Protect LaTeX replacements while ordinary LaTeX-special
    # characters in manuscript text are escaped.
    protected_tokens: dict[str, str] = {}

    for index, (char, latex) in enumerate(
        unicode_latex_replacements.items()
    ):
        if char not in value:
            continue

        token = f"@@CNSMUNICODE{index}@@"
        value = value.replace(char, token)
        protected_tokens[token] = latex

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    value = "".join(
        replacements.get(char, char)
        for char in value
    )

    for token, latex in protected_tokens.items():
        value = value.replace(token, latex)

    # A SHA-256 digest is a 64-character unbroken token. In IEEE
    # two-column layout it can exceed the column width even though the
    # digest itself is mandatory disclosure content. Add invisible
    # TeX line-break opportunities every eight hexadecimal characters.
    #
    # This is a presentation-only transformation: every hexadecimal
    # character remains unchanged and in the same order.
    sha256_pattern = re.compile(
        r"(?<![0-9A-Fa-f])"
        r"([0-9A-Fa-f]{64})"
        r"(?![0-9A-Fa-f])"
    )

    def make_sha256_breakable(
        match: re.Match[str],
    ) -> str:
        digest = match.group(1)

        return r"\allowbreak{}".join(
            digest[index:index + 8]
            for index in range(0, 64, 8)
        )

    value = sha256_pattern.sub(
        make_sha256_breakable,
        value,
    )

    return value


def _render_paragraphs(value: str) -> str:
    paragraphs = [
        item.strip()
        for item in re.split(
            r"\n\s*\n",
            value.strip(),
        )
        if item.strip()
    ]

    return "\n\n".join(
        _latex_escape(paragraph)
        for paragraph in paragraphs
    )


def _render_references(
    *,
    cited_record_ids: list[str],
    verified_records: list[dict[str, Any]],
) -> str:
    records_by_id: dict[str, dict[str, Any]] = {}

    for record in verified_records:
        for candidate_id in (
            record.get("record_id"),
            record.get("doi"),
            record.get("id"),
        ):
            if candidate_id:
                records_by_id[str(candidate_id)] = record

    rendered: list[str] = []

    for index, record_id in enumerate(
        cited_record_ids,
        1,
    ):
        record = records_by_id.get(
            str(record_id)
        )

        if record is None:
            rendered.append(
                "\\bibitem{ref"
                f"{index}"
                "} "
                + _latex_escape(
                    str(record_id)
                )
            )
            continue

        authors = record.get("authors") or []
        if isinstance(authors, list):
            author_text = ", ".join(
                str(author)
                for author in authors
            )
        else:
            author_text = str(authors)

        title = str(
            record.get("title") or record_id
        )
        venue = str(
            record.get("venue")
            or record.get("container_title")
            or record.get("source")
            or ""
        )
        year = str(
            record.get("publication_year")
            or record.get("year")
            or ""
        )
        doi = str(
            record.get("doi") or ""
        )

        parts = [
            item
            for item in (
                author_text,
                f'"{title}"',
                venue,
                year,
                doi,
            )
            if item
        ]

        rendered.append(
            "\\bibitem{ref"
            f"{index}"
            "} "
            + _latex_escape(
                ", ".join(parts)
            )
        )

    return "\n".join(rendered)


def render_ieee_latex(
    *,
    manuscript: dict[str, Any],
    verified_records: list[dict[str, Any]],
    document_class: str,
) -> str:
    sections = manuscript["sections"]

    references = _render_references(
        cited_record_ids=list(
            manuscript.get(
                "cited_record_ids",
                [],
            )
        ),
        verified_records=verified_records,
    )

    limitations = manuscript.get(
        "limitations",
        [],
    )

    limitations_text = "\n".join(
        "\\item "
        + _latex_escape(str(item))
        for item in limitations
    )

    return f"""\
{document_class}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{url}}

\\title{{{_latex_escape(manuscript["title"])}}}

\\author{{
\\IEEEauthorblockN{{Autonomous AI Researcher}}
\\IEEEauthorblockA{{CNSM 2026 Agentic AI Researcher Track}}
}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{_render_paragraphs(manuscript["abstract"])}
\\end{{abstract}}

\\section{{Introduction}}
{_render_paragraphs(sections["introduction"])}

\\section{{Related Work}}
{_render_paragraphs(sections["related_work"])}

\\section{{Methodology}}
{_render_paragraphs(sections["methodology"])}

\\section{{Results}}
{_render_paragraphs(sections["results"])}

\\section{{Discussion}}
{_render_paragraphs(sections["discussion"])}

\\section{{Limitations}}
\\begin{{itemize}}
{limitations_text}
\\end{{itemize}}

\\section{{Conclusion}}
{_render_paragraphs(sections["conclusion"])}

\\section*{{Disclosure Statement}}
{_render_paragraphs(manuscript["disclosure_statement"])}

\\begin{{thebibliography}}{{99}}
{references}
\\end{{thebibliography}}

\\end{{document}}
"""


def _pdf_page_count(pdf_path: Path) -> int:
    completed = subprocess.run(
        [
            "pdfinfo",
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(
                line.split(":", 1)[1].strip()
            )

    raise RuntimeError(
        "pdfinfo did not report a page count."
    )


def build_publication_artifacts(
    *,
    manuscript: dict[str, Any],
    verified_records: list[dict[str, Any]],
    output_dir: Path,
    paper_run_constraints: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    format_constraints = (
        paper_run_constraints["format"]
    )

    maximum_pages = int(
        format_constraints["maximum_pages"]
    )

    document_class = str(
        format_constraints[
            "latex_document_class"
        ]
    )

    tex_path = (
        output_dir / "manuscript.tex"
    )
    pdf_path = (
        output_dir / "manuscript.pdf"
    )
    log_path = (
        output_dir / "compilation.log"
    )

    latex_source = render_ieee_latex(
        manuscript=manuscript,
        verified_records=verified_records,
        document_class=document_class,
    )

    tex_path.write_text(
        latex_source,
        encoding="utf-8",
    )

    if pdf_path.exists():
        pdf_path.unlink()

    completed = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ],
        cwd=output_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_path.write_text(
        completed.stdout
        + "\n\n--- STDERR ---\n"
        + completed.stderr,
        encoding="utf-8",
    )

    compile_status = (
        "passed"
        if completed.returncode == 0
        and pdf_path.is_file()
        else "failed"
    )

    page_count: int | None = None

    if compile_status == "passed":
        page_count = _pdf_page_count(
            pdf_path
        )

    disclosure_included = (
        "\\section*{Disclosure Statement}"
        in latex_source
        and bool(
            str(
                manuscript.get(
                    "disclosure_statement",
                    "",
                )
            ).strip()
        )
    )

    references_included = (
        "\\begin{thebibliography}"
        in latex_source
    )

    disclosure_before_references = (
        "\\section*{Disclosure Statement}" in latex_source
        and "\\begin{thebibliography}" in latex_source
        and latex_source.index(
            "\\section*{Disclosure Statement}"
        )
        < latex_source.index(
            "\\begin{thebibliography}"
        )
    )

    within_page_limit = (
        page_count is not None
        and page_count <= maximum_pages
    )

    uses_full_page_budget = (
        page_count is not None
        and page_count == maximum_pages
    )

    validation = {
        "compile_status": compile_status,
        "latex_return_code": (
            completed.returncode
        ),
        "page_count": page_count,
        "maximum_pages": maximum_pages,
        "within_page_limit": (
            within_page_limit
        ),
        "uses_full_page_budget": uses_full_page_budget,
        "references_included": (
            references_included
        ),
        "references_included_in_limit": bool(
            format_constraints[
                "references_included_in_limit"
            ]
        ),
        "disclosure_included": (
            disclosure_included
        ),
        "disclosure_statement_mandatory": bool(
            format_constraints[
                "disclosure_statement_mandatory"
            ]
        ),
        "disclosure_statement_included_in_limit": bool(
            format_constraints[
                "disclosure_statement_included_in_limit"
            ]
        ),
        "disclosure_before_references": disclosure_before_references,
        "sixth_page_disclosure_prohibited": bool(
            format_constraints[
                "sixth_page_disclosure_prohibited"
            ]
        ),
        "template_manipulation_prohibited": bool(
            format_constraints[
                "template_manipulation_prohibited"
            ]
        ),
        "tex_path": str(tex_path),
        "pdf_path": (
            str(pdf_path)
            if pdf_path.is_file()
            else None
        ),
        "compilation_log_path": str(
            log_path
        ),
        "tex_sha256": _sha256_file(
            tex_path
        ),
        "pdf_sha256": (
            _sha256_file(pdf_path)
            if pdf_path.is_file()
            else None
        ),
        "compilation_log_sha256": (
            _sha256_file(log_path)
        ),
    }

    validation["passed"] = bool(
        validation["compile_status"]
        == "passed"
        and validation[
            "within_page_limit"
        ]
        and validation[
            "uses_full_page_budget"
        ]
        and validation[
            "references_included"
        ]
        and validation[
            "disclosure_included"
        ]
        and validation["disclosure_before_references"]
    )

    return validation
