"""Node 7: Grounded Question-Answering (QA) Engine.

Executes Retrieval-Augmented Generation (RAG) strictly grounded in retrieved paper chunks,
tracks provenance citations (section, page), and enforces anti-hallucination refusals.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import re
from arxiv_digest.config import AgentConfig
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import QACitation, QAResponse, TextChunk
from arxiv_digest.nodes.vector_store import LocalVectorStore, get_vector_store
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)


def format_context_chunks(chunks_with_scores: list[tuple[TextChunk, float]]) -> tuple[str, list[QACitation]]:
    """Format retrieved chunks with clear source attribution and create citation models."""
    formatted_texts = []
    citations: list[QACitation] = []

    for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
        formatted_texts.append(
            f"[Source {idx}] (Section: {chunk.section_heading}, Page: {chunk.page_number}, Relevance: {score:.2f}):\n"
            f"{chunk.text}\n"
        )
        # Extract a representative 100-character excerpt
        excerpt = chunk.text[:120].strip().replace("\n", " ") + "..."
        citations.append(
            QACitation(
                chunk_id=chunk.chunk_id,
                section=chunk.section_heading,
                page=chunk.page_number,
                excerpt=excerpt,
                relevance_score=round(score, 3),
            )
        )

    return "\n".join(formatted_texts), citations


def answer_question(
    question: str,
    state: AgentState,
    llm: BaseLLM,
    config: AgentConfig,
) -> QAResponse:
    """Answer a user question strictly grounded in the indexed paper."""
    # Retrieve active vector store
    store = get_vector_store(state.session_id)
    if not store:
        if state.chunks:
            store = LocalVectorStore(chunks=state.chunks)
        else:
            return QAResponse(
                question=question,
                answer="No paper chunks are currently loaded in the vector store.",
                citations=[],
                is_grounded=False,
                confidence_score=0.0,
            )

    # Vector search with threshold
    retrieved = store.search(
        query=question,
        top_k=config.retrieval_top_k,
        min_threshold=config.min_similarity_threshold,
    )

    # Anti-Hallucination Guard: If no chunks meet minimum similarity threshold
    if not retrieved:
        return QAResponse(
            question=question,
            answer=(
                "I cannot answer this question based on the paper. The document does not contain "
                "relevant information regarding this query, and answers are strictly restricted "
                "to verified source content to prevent hallucination."
            ),
            citations=[],
            is_grounded=False,
            confidence_score=0.0,
        )

    context_str, citations = format_context_chunks(retrieved)
    paper_title = state.selected_paper.title if state.selected_paper else "the paper"

    system_prompt = (
        "You are an objective research assistant answering questions about a scientific paper. "
        "You MUST answer strictly and exclusively based on the provided Context Sources. "
        "Every single factual assertion must cite its source (e.g., '[Source 1]' or '[Section 3, p. 4]'). "
        "If the Context does not contain the answer, explicitly refuse and state that the paper does not mention it. "
        "Never extrapolate or hallucinate outside the retrieved text."
    )

    user_prompt = f"""PAPER: {paper_title}

CONTEXT SOURCES:
{context_str}

USER QUESTION:
{question}

Provide a precise, grounded answer citing specific sources. If the answer is absent from the sources, state so clearly:"""

    try:
        answer_text = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )
    except Exception as e:
        logger.warning(f"LLM generation failed ({e}). Returning fallback response.")
        answer_text = f"An error occurred while synthesizing the answer: {e}"

    # Determine if response refused or indicated absence
    refusal_keywords = [
        "does not mention",
        "not covered in the paper",
        "cannot answer this question based on the paper",
        "not contain sufficient information",
    ]
    is_grounded = not any(kw in answer_text.lower() for kw in refusal_keywords)

    response = QAResponse(
        question=question,
        answer=answer_text.strip(),
        citations=citations if is_grounded else [],
        is_grounded=is_grounded,
        confidence_score=round(float(retrieved[0][1]), 3) if retrieved else 0.0,
    )

    state.record_qa_interaction(response)
    return response
