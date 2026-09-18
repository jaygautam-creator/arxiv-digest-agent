"""Node 5: Section-Aware Semantic Text Chunker.

Implements structural, section-bounded chunking with sliding sentence overlap
and provenance tracking (section heading, page number).

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import re
import time
from arxiv_digest.config import AgentConfig
from arxiv_digest.models import ParsedPaper, TextChunk
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
) -> list[TextChunk]:
    """Generate overlapping semantic chunks bounded strictly within section scope."""
    chunks: list[TextChunk] = []
    if not content.strip():
        return chunks

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    current_chunk_text = ""
    current_sentences: list[str] = []
    chunk_counter = 0

    # Approximate page number interpolation
    num_paras = max(1, len(paragraphs))

    for p_idx, para in enumerate(paragraphs):
        approx_page = page_start + int((p_idx / num_paras) * (page_end - page_start))
        sentences = split_into_sentences(para)

        for sent in sentences:
            test_chunk = (current_chunk_text + " " + sent).strip()
            if len(test_chunk) <= chunk_size:
                current_chunk_text = test_chunk
                current_sentences.append(sent)
            else:
                if current_chunk_text:
                    chunk_counter += 1
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{heading[:15]}_{chunk_counter}",
                            section_heading=heading,
                            page_number=approx_page,
                            text=current_chunk_text,
                            token_count=len(current_chunk_text.split()),
                        )
                    )

                # Keep overlap sentences from the tail
                overlap_text = ""
                retained_sentences: list[str] = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) < chunk_overlap:
                        overlap_text = s + " " + overlap_text
                        retained_sentences.insert(0, s)
                    else:
                        break

                current_sentences = retained_sentences + [sent]
                current_chunk_text = " ".join(current_sentences)

    # Emit final chunk
    if current_chunk_text.strip():
        chunk_counter += 1
        chunks.append(
            TextChunk(
                chunk_id=f"{heading[:15]}_{chunk_counter}",
                section_heading=heading,
                page_number=page_end,
                text=current_chunk_text,
                token_count=len(current_chunk_text.split()),
            )
        )

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

    for section in parsed.sections:
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
