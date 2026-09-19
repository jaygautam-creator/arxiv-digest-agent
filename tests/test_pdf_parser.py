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


def test_table_rows_rebuilt_with_column_headers():
    from arxiv_digest.nodes.pdf_parser import Geometry, _render_table, _table_rows

    def g(block, y, x0, x1, text):
        return Geometry(block, y, y + 8, x0, x1, text)

    # Header row in its own block above the data; each data cell is its own PDF line,
    # and two numeric cells were merged into one line by the PDF generator.
    header = [
        g(1, 90, 60, 80, "Full"),
        g(1, 90, 95, 115, "Ours"),
        g(1, 90, 130, 170, "Ratio"),
        g(1, 78, 55, 120, "KV Cache"),
    ]
    data = [g(2, 100, 10, 50, "LLaMA-3-8B"), g(2, 100.4, 62, 75, "8G"), g(2, 100.2, 97, 150, "4.8G 60%")]
    rows = _table_rows(data)
    assert [[text for _, text in row] for row in rows] == [["LLaMA-3-8B", "8G", "4.8G", "60%"]]
    assert _render_table(rows, 100, header + data, block=2) == ["LLaMA-3-8B | Full: 8G | Ours: 4.8G | Ratio: 60%"]


def test_prose_block_is_not_a_table():
    from arxiv_digest.nodes.pdf_parser import Geometry, _table_rows

    prose = [
        Geometry(1, y, y + 8, 10, 300, t)
        for y, t in [(100, "GRKV raises the average"), (112, "score from 27.44 to 29.09"), (124, "with SnapKV.")]
    ]
    assert _table_rows(prose) is None


def _paper() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2401.00001", title="t", abstract="a",
        pdf_url="https://arxiv.org/pdf/2401.00001", abs_url="u", published_date="2024-01-01",
    )  # fmt: skip


def test_download_rejects_html_and_replaces_a_bad_cached_file(tmp_path):
    from io import BytesIO
    from unittest.mock import patch

    from arxiv_digest.nodes.pdf_parser import download_pdf

    cached = tmp_path / "2401.00001.pdf"
    cached.write_bytes(b"<html>rate limited</html>" * 100)  # a bad file left by an earlier run

    html = BytesIO(b"<html>Too many requests</html>" * 100)
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: html):
        assert download_pdf(_paper(), tmp_path) is None
    assert not cached.exists()

    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: BytesIO(b"%PDF-1.7 real content")):
        assert download_pdf(_paper(), tmp_path) == cached
    assert cached.read_bytes().startswith(b"%PDF-")
