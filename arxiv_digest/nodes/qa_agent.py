"""Node 7: Grounded Question-Answering (QA) Engine.

Executes Retrieval-Augmented Generation (RAG) strictly grounded in retrieved paper chunks,
tracks provenance citations (section, page), and enforces anti-hallucination refusals.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import time

from arxiv_digest.config import AgentConfig
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import QACitation, QAResponse, TextChunk
from arxiv_digest.nodes.chunker import metadata_chunk
from arxiv_digest.nodes.vector_store import build_vector_store, get_vector_store, register_vector_store, retrieve
from arxiv_digest.state import AgentState, StepStatus

logger = logging.getLogger(__name__)

# The model is told to open any refusal with this marker, which is more reliable than
# guessing from free-form wording. REFUSAL_PHRASES remains as a backstop.
REFUSAL_MARKER = "NOT IN PAPER:"
REFUSAL_PHRASES = (
    "does not mention",
    "not covered in the paper",
    "cannot answer this question based on the",
    "not contain sufficient information",
    "the provided sources do not",
    "the context does not",
    "excerpts do not",
    "does not specify",
    "does not address",
)


def _strip_marker(answer: str) -> str | None:
    """Return the text after a leading refusal marker (tolerating Markdown emphasis), else None."""
    text = answer.strip().lstrip("*_ ")
    if not text.upper().startswith(REFUSAL_MARKER):
        return None
    return text[len(REFUSAL_MARKER) :].lstrip("*_ ").strip()


def is_refusal(answer: str) -> bool:
    """True when the model reported that the retrieved sources do not answer the question."""
    return _strip_marker(answer) is not None or any(p in answer.lower() for p in REFUSAL_PHRASES)


def format_context_chunks(chunks_with_scores: list[tuple[TextChunk, float]]) -> tuple[str, list[QACitation]]:
    """Format retrieved chunks with clear source attribution and create citation models."""
    formatted_texts = []
    citations: list[QACitation] = []

    for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
        formatted_texts.append(
            f"[Source {idx}] (Section: {chunk.section_heading}, Page: {chunk.page_label}, Relevance: {score:.2f}):\n"
            f"{chunk.text}\n"
        )
        # Extract a representative 100-character excerpt
        excerpt = chunk.text[:120].strip().replace("\n", " ") + "..."
        citations.append(
            QACitation(
                chunk_id=chunk.chunk_id,
                section=chunk.section_heading,
                page=chunk.page_number,
                page_label=chunk.page_label,
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
            # Resumed session: rebuild the index from persisted chunks once, then reuse it.
            # Sessions saved by older versions lack the metadata chunk, so add it here.
            if state.selected_paper and not any(c.chunk_id == "metadata" for c in state.chunks):
                state.chunks.insert(0, metadata_chunk(state.selected_paper))
            store = build_vector_store(state.chunks, config)
            register_vector_store(state.session_id, store)
        else:
            return QAResponse(
                question=question,
                answer="No paper chunks are currently loaded in the vector store.",
                citations=[],
                is_grounded=False,
                confidence_score=0.0,
            )

    # Vector search with threshold
    retrieved = retrieve(store, question, config)

    # Anti-Hallucination Guard: If no chunks meet minimum similarity threshold
    if not retrieved:
        response = QAResponse(
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
        state.record_qa_interaction(response)
        return response

    context_str, citations = format_context_chunks(retrieved)
    paper_title = state.selected_paper.title if state.selected_paper else "the paper"

    system_prompt = (
        "You are an objective research assistant answering questions about a scientific paper. "
        "You MUST answer strictly and exclusively based on the provided Context Sources. "
        "Every single factual assertion must cite its source (e.g., '[Source 1]' or '[Section 3, p. 4]'). "
        f"If the Context does not contain the answer, begin your reply with '{REFUSAL_MARKER}' and explain in one "
        "sentence what is missing. "
        "Never extrapolate or hallucinate outside the retrieved text. "
        "The context is extracted from a PDF. Tables appear as one row per line, cells separated by ' | ', with "
        "the row label first; column headers may be missing. Quote a table value only when its row and column "
        "are unambiguous from the surrounding text, and name the row, model and benchmark; otherwise say the "
        "table could not be read reliably. Prefer numbers stated in prose."
    )

    user_prompt = f"""PAPER: {paper_title}

CONTEXT SOURCES:
{context_str}

USER QUESTION:
{question}

Provide a precise, grounded answer citing specific sources.
If the answer is absent from the sources, state so clearly:"""

    try:
        answer_text = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )
    except Exception as e:
        logger.warning(f"LLM generation failed ({e}).")
        response = QAResponse(
            question=question,
            answer=f"The language model could not be reached, so no answer was generated ({e}).",
            citations=[],
            is_grounded=False,
            confidence_score=0.0,
        )
        state.record_qa_interaction(response)
        return response

    is_grounded = not is_refusal(answer_text)
    explanation = _strip_marker(answer_text)
    if explanation is not None:
        answer_text = f"The retrieved passages do not answer this. {explanation}"

    response = QAResponse(
        question=question,
        answer=answer_text.strip(),
        citations=citations if is_grounded else [],
        is_grounded=is_grounded,
        confidence_score=round(float(retrieved[0][1]), 3) if retrieved else 0.0,
    )

    state.record_qa_interaction(response)
    return response


def answer_question_node(state: AgentState, llm: BaseLLM, config: AgentConfig) -> AgentState:
    """Graph node: answer `state.pending_question` and append the turn to `state.qa_history`."""
    start_time = time.time()
    question = (state.pending_question or "").strip()
    state.pending_question = None
    if not question:
        state.add_error("answer_question ran without a pending question.")
        state.log_step("answer_question", "error", "No question provided.", (time.time() - start_time) * 1000)
        return state
    if not state.chunks:
        state.add_error("No indexed paper in this session; run an analysis first.")
        state.log_step("answer_question", "error", "No chunks to search.", (time.time() - start_time) * 1000)
        return state

    response = answer_question(question, state, llm, config)
    status: StepStatus = "success" if response.is_grounded else "warning"
    outcome = "Grounded answer" if response.is_grounded else "Not answered from the paper"
    msg = f"{outcome} ({len(response.citations)} citations)."
    state.log_step("answer_question", status, msg, (time.time() - start_time) * 1000)
    return state
