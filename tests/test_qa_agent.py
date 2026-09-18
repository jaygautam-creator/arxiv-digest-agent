"""Tests for QA agent grounding & hallucination refusal.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.config import AgentConfig
from arxiv_digest.llm.mock_client import MockLLM
from arxiv_digest.models import PaperMetadata, TextChunk
from arxiv_digest.nodes.qa_agent import answer_question
from arxiv_digest.nodes.vector_store import LocalVectorStore, register_vector_store
from arxiv_digest.state import AgentState


def test_grounded_qa_with_citations():
    paper = PaperMetadata(
        arxiv_id="2401.12345",
        title="Sparse Attention Mechanisms",
        authors=["Alice Smith"],
        abstract="A study on sparse attention.",
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        abs_url="https://arxiv.org/abs/2401.12345",
        published_date="2024-01-01",
    )
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            section_heading="Results",
            page_number=5,
            text="Our sparse attention method achieves 3.5x speedup with less than 0.1% perplexity drop.",
        )
    ]
    state = AgentState(
        session_id="test_session_1",
        selected_paper=paper,
        chunks=chunks,
    )
    store = LocalVectorStore(chunks=chunks)
    register_vector_store(state.session_id, store)

    llm = MockLLM()
    config = AgentConfig()

    response = answer_question("How much speedup does the sparse attention achieve?", state, llm, config)
    assert response.is_grounded is True
    assert len(response.citations) > 0
    assert response.citations[0].page == 5
    assert "Results" in response.citations[0].section


def test_anti_hallucination_refusal():
    paper = PaperMetadata(
        arxiv_id="2401.12345",
        title="Sparse Attention Mechanisms",
        authors=["Alice Smith"],
        abstract="A study on sparse attention.",
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        abs_url="https://arxiv.org/abs/2401.12345",
        published_date="2024-01-01",
    )
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            section_heading="Results",
            page_number=5,
            text="Our sparse attention method achieves 3.5x speedup with less than 0.1% perplexity drop.",
        )
    ]
    state = AgentState(
        session_id="test_session_2",
        selected_paper=paper,
        chunks=chunks,
    )
    store = LocalVectorStore(chunks=chunks)
    register_vector_store(state.session_id, store)

    llm = MockLLM()
    config = AgentConfig()

    # Query completely unrelated to sparse attention
    response = answer_question("What is the capital of France and what is the weather there?", state, llm, config)
    assert response.is_grounded is False
    assert len(response.citations) == 0
    assert "cannot answer" in response.answer.lower() or "not contain" in response.answer.lower()


def test_refusal_detection_uses_marker_and_phrases():
    from arxiv_digest.nodes.qa_agent import is_refusal

    assert is_refusal("NOT IN PAPER: the sources never discuss fine-tuning.")
    assert is_refusal("**NOT IN PAPER:** the sources never discuss fine-tuning.")
    assert is_refusal("The provided excerpts do not contain any statement about this.")
    assert not is_refusal("GRKV raises the average score from 27.44 to 29.09 [Source 2].")
