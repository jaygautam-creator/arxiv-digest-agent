"""End-to-end tests for the Research State Graph.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.graph import Context, GraphError, ResearchStateGraph, StateGraph, build_research_graph
from arxiv_digest.llm.mock_client import MockLLM
from arxiv_digest.models import PaperMetadata
from arxiv_digest.state import AgentState


def dummy_fetch(*args, **kwargs):
    return [
        PaperMetadata(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract=(
                "The dominant sequence transduction models are based on complex recurrent "
                "or convolutional neural networks..."
            ),
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            abs_url="https://arxiv.org/abs/1706.03762",
            published_date="2017-06-12",
        )
    ]


@patch("arxiv_digest.nodes.arxiv_client.fetch_from_arxiv", side_effect=dummy_fetch)
@patch("arxiv_digest.nodes.pdf_parser.download_pdf", return_value=None)
def test_full_graph_execution(mock_download, mock_arxiv, tmp_path):
    config = AgentConfig(
        provider=LLMProviderType.MOCK,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        sessions_dir=tmp_path / "sessions",
    )
    llm = MockLLM()
    graph = ResearchStateGraph(config=config, llm=llm)

    state = graph.execute(query="1706.03762")

    assert state.intent == "DIRECT_ID"
    assert state.selected_paper is not None
    assert state.selected_paper.arxiv_id == "1706.03762"
    assert len(state.chunks) > 0
    assert state.briefing is not None
    assert state.briefing.title == "Attention Is All You Need"
    assert len(state.briefing.method_approach) > 0
    assert len(state.briefing.limitations) > 0
    assert state.is_complete is True

    # Check that execution logs captured each node
    logged_nodes = [log.node_name for log in state.execution_logs]
    assert "query_understanding" in logged_nodes
    assert "arxiv_retrieval" in logged_nodes
    assert "fetch_and_parse" in logged_nodes
    assert "chunk_and_embed" in logged_nodes
    assert "vector_indexing" in logged_nodes
    assert "summarize_briefing" in logged_nodes


def _config(tmp_path):
    return AgentConfig(
        provider=LLMProviderType.MOCK,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        sessions_dir=tmp_path / "sessions",
    )


@patch("arxiv_digest.nodes.arxiv_client.fetch_from_arxiv", side_effect=dummy_fetch)
@patch("arxiv_digest.nodes.pdf_parser.download_pdf", return_value=None)
def test_direct_id_skips_ranking_and_topic_search_uses_it(mock_download, mock_arxiv, tmp_path):
    graph = ResearchStateGraph(config=_config(tmp_path), llm=MockLLM())

    direct = graph.execute("1706.03762")
    topic = graph.execute("attention mechanisms for translation")

    common = ["fetch_and_parse", "chunk_and_embed", "vector_indexing", "summarize_briefing", "persist_session"]
    assert direct.visited_nodes == ["query_understanding", "arxiv_retrieval", *common]
    assert topic.visited_nodes == ["query_understanding", "arxiv_retrieval", "paper_ranking", *common]
    assert direct.current_node == topic.current_node == "end"


@patch("arxiv_digest.nodes.arxiv_client.fetch_from_arxiv", return_value=[])
def test_node_error_routes_to_error_state(mock_arxiv, tmp_path):
    state = ResearchStateGraph(config=_config(tmp_path), llm=MockLLM()).execute("2401.99999")

    assert state.current_node == "error"
    assert state.visited_nodes == ["query_understanding", "arxiv_retrieval"]
    assert state.errors and state.briefing is None


@patch("arxiv_digest.nodes.arxiv_client.fetch_from_arxiv", side_effect=dummy_fetch)
@patch("arxiv_digest.nodes.pdf_parser.download_pdf", return_value=None)
def test_ask_entry_point_answers_and_persists(mock_download, mock_arxiv, tmp_path):
    config = _config(tmp_path)
    graph = ResearchStateGraph(config=config, llm=MockLLM())
    state = graph.execute("1706.03762")

    state = graph.ask(state, "What are sequence transduction models based on?")

    assert state.visited_nodes[-2:] == ["answer_question", "persist_session"]
    assert len(state.qa_history) == 1 and state.pending_question is None
    saved = AgentState.load_session(config.sessions_dir / f"session_{state.session_id}.json")
    assert len(saved.qa_history) == 1


def test_ask_without_question_is_an_error(tmp_path):
    state = ResearchStateGraph(config=_config(tmp_path), llm=MockLLM()).ask(AgentState(), "  ")
    assert state.current_node == "error"


def test_validate_rejects_bad_graphs():
    noop = lambda s, c: s  # noqa: E731
    with pytest.raises(GraphError, match="unknown node"):
        StateGraph().add_node("a", noop, "").add_edge("a", "missing").validate()
    with pytest.raises(GraphError, match="no outgoing edge"):
        StateGraph().add_node("a", noop, "").validate()
    with pytest.raises(GraphError, match="unconditional fallback"):
        StateGraph().add_node("a", noop, "").add_edge("a", "end", lambda s: False).validate()


def test_cycle_guard_stops_runaway_graphs(tmp_path):
    loop = StateGraph().add_node("a", lambda s, c: s, "").add_edge("a", "a").add_entry_point("start", "a")
    with pytest.raises(GraphError, match="cycle"):
        loop.run(AgentState(), Context(_config(tmp_path), MockLLM()), "start", max_steps=5)


def test_readme_diagram_is_generated_from_the_graph():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert build_research_graph().to_mermaid() in readme, (
        "Regenerate the README diagram: python -m arxiv_digest --graph"
    )


@patch("arxiv_digest.nodes.arxiv_client.fetch_from_arxiv", side_effect=dummy_fetch)
@patch("arxiv_digest.nodes.pdf_parser.download_pdf", return_value=None)
def test_saved_session_includes_the_persist_step(mock_download, mock_arxiv, tmp_path):
    config = _config(tmp_path)
    state = ResearchStateGraph(config=config, llm=MockLLM()).execute("1706.03762")

    saved = AgentState.load_session(config.sessions_dir / f"session_{state.session_id}.json")
    assert saved.visited_nodes[-1] == "persist_session"
