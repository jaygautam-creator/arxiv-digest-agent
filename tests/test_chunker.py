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
