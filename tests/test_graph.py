"""End-to-end tests for the Research State Graph.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from unittest.mock import patch
from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.graph import ResearchStateGraph
from arxiv_digest.llm.mock_client import MockLLM
from arxiv_digest.models import PaperMetadata


def dummy_fetch(*args, **kwargs):
    return [
        PaperMetadata(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
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
    assert "query_parser" in logged_nodes
    assert "arxiv_retrieval" in logged_nodes
    assert "fetch_and_parse" in logged_nodes
    assert "chunk_and_embed" in logged_nodes
    assert "vector_indexing" in logged_nodes
    assert "summarize_briefing" in logged_nodes
