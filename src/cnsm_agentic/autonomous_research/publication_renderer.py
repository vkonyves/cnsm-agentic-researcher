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
    unicode_replacements = {
        "\u2013": "--",     # en dash
        "\u2014": "---",    # em dash
        "\u2212": "-",      # mathematical minus sign
        "\u00a0": " ",      # non-breaking space
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2265": ">=",
        "\u2264": "<=",
    }

    value = "".join(
        unicode_replacements.get(char, char)
        for char in value
    )

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

    return "".join(
        replacements.get(char, char)
        for char in value
    )


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

    within_page_limit = (
        page_count is not None
        and page_count <= maximum_pages
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
            "references_included"
        ]
        and validation[
            "disclosure_included"
        ]
    )

    return validation
