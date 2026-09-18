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
