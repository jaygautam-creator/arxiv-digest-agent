"""Tests for section-aware text chunking.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

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
