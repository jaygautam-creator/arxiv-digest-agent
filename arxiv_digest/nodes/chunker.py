"""Node 5: Section-Aware Semantic Text Chunker.

Implements structural, section-bounded chunking with sliding sentence overlap
and provenance tracking (section heading, page number).

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import re
import time

from arxiv_digest.config import AgentConfig
from arxiv_digest.models import TextChunk
from arxiv_digest.state import AgentState


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common scientific abbreviations."""
    # Simple regex splitting on punctuation followed by whitespace and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks_for_section(
    heading: str,
    content: str,
    page_start: int,
    page_end: int,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    section_index: int = 0,
    paragraph_pages: list[int] | None = None,
) -> list[TextChunk]:
    """Pack sentences into overlapping chunks that never cross the section boundary.

    Each chunk is labelled with the page its first sentence comes from. Exact pages come
    from `paragraph_pages` (one per paragraph, recorded by the PDF parser); without them,
    the page is interpolated across the section's page range.
    """
    if not content.strip():
        return []

    raw_paragraphs = content.split("\n\n")
    if paragraph_pages is None or len(paragraph_pages) != len(raw_paragraphs):
        span = max(1, len(raw_paragraphs))
        paragraph_pages = [page_start + (i * (page_end - page_start)) // span for i in range(len(raw_paragraphs))]

    # Units are sentences, or whole rows for reconstructed tables ("label | v1 | v2"),
    # which must stay on their own lines so values remain attached to their row.
    units: list[tuple[str, int, bool]] = []
    for paragraph, page in zip(raw_paragraphs, paragraph_pages, strict=True):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if " | " in paragraph:
            units.extend((row.strip(), page, " | " in row) for row in paragraph.splitlines() if row.strip())
        else:
            units.extend((sentence, page, False) for sentence in split_into_sentences(paragraph))

    chunks: list[TextChunk] = []
    current: list[tuple[str, int, bool]] = []

    def emit() -> None:
        text = current[0][0]
        for unit, _, is_row in current[1:]:
            text += ("\n" if is_row else " ") + unit
        chunks.append(
            TextChunk(
                chunk_id=f"s{section_index}_c{len(chunks) + 1}",
                section_heading=heading,
                page_number=current[0][1],
                page_end=current[-1][1],
                text=text,
                token_count=len(text.split()),
            )
        )

    for unit in units:
        candidate_length = sum(len(u) + 1 for u, _, _ in current) + len(unit[0])
        if current and candidate_length > chunk_size:
            emit()
            # Carry trailing units (up to chunk_overlap characters) into the next chunk.
            overlap: list[tuple[str, int, bool]] = []
            for item in reversed(current):
                if sum(len(u) + 1 for u, _, _ in overlap) + len(item[0]) >= chunk_overlap:
                    break
                overlap.insert(0, item)
            current = overlap
        current.append(unit)

    if current:
        emit()
    return chunks


def chunk_and_embed_node(state: AgentState, config: AgentConfig) -> AgentState:
    """Graph Node: Segment parsed paper into structured chunks with provenance."""
    start_time = time.time()

    if state.parsed_paper is None:
        msg = "No parsed paper available for chunking."
        state.add_error(msg)
        state.log_step("chunk_and_embed", "error", msg, (time.time() - start_time) * 1000)
        return state

    parsed = state.parsed_paper
    all_chunks: list[TextChunk] = []

    for section_index, section in enumerate(parsed.sections):
        # Don't create chunks for references to avoid polluting technical retrieval
        if "reference" in section.heading.lower() or "bibliography" in section.heading.lower():
            continue

        section_chunks = create_chunks_for_section(
            heading=section.heading,
            content=section.content,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            section_index=section_index,
            paragraph_pages=section.paragraph_pages,
        )
        all_chunks.extend(section_chunks)

    # Edge case: if no chunks created (e.g. abstract only), create single chunk
    if not all_chunks and parsed.metadata.abstract:
        all_chunks.append(
            TextChunk(
                chunk_id="abstract_1",
                section_heading="Abstract",
                page_number=1,
                text=parsed.metadata.abstract,
                token_count=len(parsed.metadata.abstract.split()),
            )
        )

    state.chunks = all_chunks
    msg = f"Generated {len(all_chunks)} section-aware chunks across {len(parsed.sections)} sections."
    state.log_step("chunk_and_embed", "success", msg, (time.time() - start_time) * 1000)

    return state
