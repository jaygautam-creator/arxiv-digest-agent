"""Tests for PDF parser & section fallback.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from pathlib import Path
from arxiv_digest.models import PaperMetadata
from arxiv_digest.nodes.pdf_parser import parse_pdf_document


def test_pdf_fallback_to_abstract():
    paper = PaperMetadata(
        arxiv_id="2401.99999",
        title="Unpublished Conceptual Study",
        authors=["Researcher A"],
        abstract="This paper introduces a novel hypothesis regarding neural scaling laws.",
        pdf_url="https://arxiv.org/pdf/2401.99999.pdf",
        abs_url="https://arxiv.org/abs/2401.99999",
        published_date="2024-01-01",
    )

    # Pass non-existent PDF path
    parsed = parse_pdf_document(Path("/non/existent/path.pdf"), paper)

    assert parsed.metadata.arxiv_id == "2401.99999"
    assert len(parsed.sections) >= 1
    assert parsed.sections[0].heading == "Abstract"
    assert "novel hypothesis" in parsed.sections[0].content
    assert parsed.parse_warning is not None


def _write_pdf(path: Path) -> None:
    """Build a small LaTeX-style PDF: bold headings, number and title on separate lines."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    y = 72

    def put(text: str, bold: bool = False, size: float = 10) -> None:
        nonlocal y
        page.insert_text((72, y), text, fontname="hebo" if bold else "helv", fontsize=size)
        y += size + 8

    put("Abstract", bold=True, size=12)
    put("We study retrieval for long documents.")
    put("1", bold=True, size=12)
    put("Introduction", bold=True, size=12)
    put("Long documents are hard to search.")
    put("3.2 Ridge Regression for", bold=True)
    put("Cache Merging", bold=True)
    put("We solve a ridge regression problem.")
    put("References", bold=True, size=12)
    put("[1] A. Author. A paper. 2024.")
    put("[2] B. Author. Another paper. 2025.")
    doc.save(str(path))


def test_pymupdf_detects_numbered_and_wrapped_headings(tmp_path):
    from arxiv_digest.nodes.pdf_parser import extract_with_pymupdf

    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf)
    sections, references, _ = extract_with_pymupdf(pdf)
    headings = [s.heading for s in sections]

    assert "Abstract" in headings
    assert "1 Introduction" in headings
    assert "1 Introduction › 3.2 Ridge Regression for Cache Merging" in headings
    assert len(references) == 2 and references[0].startswith("[1]")
