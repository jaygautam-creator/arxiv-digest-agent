"""Tests for section-aware text chunking.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.models import TextChunk
from arxiv_digest.nodes.chunker import create_chunks_for_section


def test_section_aware_chunking():
    heading = "Methodology"
    content = (
        "We introduce a dynamic pruning policy that evaluates token importance at every step. "
        "By discarding uninformative key-value pairs, the memory footprint drops drastically.\n\n"
        "In our second stage, we compress the remaining tokens using low-rank factorization. "
        "This maintains high attention precision across multi-head attention layers."
    )

    chunks = create_chunks_for_section(
        heading=heading,
        content=content,
        page_start=3,
        page_end=4,
        chunk_size=150,
        chunk_overlap=30,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.section_heading == heading
        assert chunk.page_number in (3, 4)
        assert len(chunk.text) > 0


def test_chunks_carry_exact_page_ranges():
    from arxiv_digest.nodes.chunker import create_chunks_for_section

    paragraphs = [
        "First page sentence one. First page sentence two.",
        "Second page text follows here.",
        "Third page ends it.",
    ]
    chunks = create_chunks_for_section(
        heading="3 Method",
        content="\n\n".join(paragraphs),
        page_start=4,
        page_end=6,
        chunk_size=60,
        chunk_overlap=0,
        paragraph_pages=[4, 5, 6],
    )
    assert [(c.page_number, c.page_end) for c in chunks] == [(4, 4), (5, 6)]
    assert chunks[1].page_label == "5–6"


def test_table_rows_stay_on_their_own_lines():
    from arxiv_digest.nodes.chunker import create_chunks_for_section

    content = "Table 2 reports RULER scores.\n\nSnapKV | 3.24 | 27.44\nw. GRKV | 1.78 | 29.09"
    chunks = create_chunks_for_section("4 Experiments", content, 7, 7)
    assert chunks[0].text == "Table 2 reports RULER scores.\nSnapKV | 3.24 | 27.44\nw. GRKV | 1.78 | 29.09"


def test_metadata_chunk_makes_authors_and_date_retrievable():
    from arxiv_digest.models import PaperMetadata
    from arxiv_digest.nodes.chunker import metadata_chunk
    from arxiv_digest.nodes.vector_store import LocalVectorStore

    paper = PaperMetadata(
        arxiv_id="2605.31105",
        title="GRKV",
        authors=["Junjie Peng", "You Wu"],
        abstract="a",
        pdf_url="u",
        abs_url="https://arxiv.org/abs/2605.31105",
        published_date="2026-05-29T10:16:30Z",
    )
    body = TextChunk(
        chunk_id="s1_c1", section_heading="3 Methods", page_number=3, text="Ridge regression merges tokens."
    )
    store = LocalVectorStore(chunks=[metadata_chunk(paper), body])

    for question in ("who is author", "When was it published?", "What is the arXiv ID?"):
        hits = store.search(question, min_threshold=0.05)
        assert hits and hits[0][0].chunk_id == "metadata", question
    assert "Junjie Peng, You Wu" in store.chunks[0].text and "2026-05-29" in store.chunks[0].text
