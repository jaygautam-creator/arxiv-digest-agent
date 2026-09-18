"""Stateful Agent Graph Orchestrator.

Implements an explicit stateful graph with nodes, conditional edges,
error boundaries, and shared state passing.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import time
from typing import Callable
from arxiv_digest.config import AgentConfig
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.nodes.arxiv_client import arxiv_retrieval_node
from arxiv_digest.nodes.chunker import chunk_and_embed_node
from arxiv_digest.nodes.pdf_parser import fetch_and_parse_node
from arxiv_digest.nodes.query_parser import parse_query_node
from arxiv_digest.nodes.ranker import paper_ranking_node
from arxiv_digest.nodes.summarizer import summarize_briefing_node
from arxiv_digest.nodes.vector_store import index_chunks_node
from arxiv_digest.state import AgentState

logger = logging.getLogger(__name__)

# Callback type for real-time progress updates: (node_name, status, message)
ProgressCallback = Callable[[str, str, str], None]


class ResearchStateGraph:
    """Explicit stateful graph coordinating the arXiv digest pipeline."""

    def __init__(self, config: AgentConfig, llm: BaseLLM):
        self.config = config
        self.llm = llm

    def execute(
        self,
        query: str,
        on_progress: ProgressCallback | None = None,
    ) -> AgentState:
        """Execute the state graph end-to-end starting from user query."""
        state = AgentState(raw_query=query)

        def _notify(node: str, status: str, msg: str):
            if on_progress:
                on_progress(node, status, msg)

        # -------------------------------------------------------------
        # Node 1: Query Understanding
        # -------------------------------------------------------------
        _notify("query_understanding", "running", f"Parsing query intent for: '{query}'")
        state = parse_query_node(state)
        if state.errors:
            _notify("query_understanding", "error", state.errors[-1])
            return state
        _notify("query_understanding", "success", f"Intent: {state.intent}")

        # -------------------------------------------------------------
        # Node 2: arXiv Retrieval
        # -------------------------------------------------------------
        _notify("arxiv_retrieval", "running", "Querying official arXiv Atom API...")
        state = arxiv_retrieval_node(state, self.config)
        if state.errors:
            _notify("arxiv_retrieval", "error", state.errors[-1])
            return state
        _notify("arxiv_retrieval", "success", f"Retrieved {len(state.candidate_papers)} paper(s)")

        # -------------------------------------------------------------
        # Conditional Edge: Topic Search vs Direct ID
        # -------------------------------------------------------------
        if state.intent == "TOPIC_SEARCH" and len(state.candidate_papers) > 1:
            # Node 3: Candidate Selection / Ranking
            _notify("paper_ranking", "running", "Ranking candidates by relevance & recency...")
            state = paper_ranking_node(state, self.llm)
            if state.errors:
                _notify("paper_ranking", "error", state.errors[-1])
                return state
            _notify("paper_ranking", "success", f"Selected: {state.selected_paper.title if state.selected_paper else 'Unknown'}")
        else:
            if not state.selected_paper and state.candidate_papers:
                state.selected_paper = state.candidate_papers[0]

        # -------------------------------------------------------------
        # Node 4: Fetch & Parse PDF
        # -------------------------------------------------------------
        paper_title = state.selected_paper.title if state.selected_paper else "target"
        _notify("fetch_and_parse", "running", f"Downloading & parsing PDF for: '{paper_title[:40]}...'")
        state = fetch_and_parse_node(state, self.config)
        if state.errors:
            _notify("fetch_and_parse", "error", state.errors[-1])
            return state
        _notify("fetch_and_parse", "success", f"Parsed {len(state.parsed_paper.sections)} sections")

        # -------------------------------------------------------------
        # Node 5: Chunk Text
        # -------------------------------------------------------------
        _notify("chunk_and_embed", "running", "Performing section-aware semantic chunking...")
        state = chunk_and_embed_node(state, self.config)
        if state.errors:
            _notify("chunk_and_embed", "error", state.errors[-1])
            return state
        _notify("chunk_and_embed", "success", f"Generated {len(state.chunks)} chunks")

        # -------------------------------------------------------------
        # Node 6: Local Vector Store Indexing
        # -------------------------------------------------------------
        _notify("vector_indexing", "running", "Indexing chunks into LocalVectorStore...")
        state = index_chunks_node(state, self.config)
        if state.errors:
            _notify("vector_indexing", "error", state.errors[-1])
            return state
        _notify("vector_indexing", "success", "Vector index active & persisted")

        # -------------------------------------------------------------
        # Node 7: Summarize Executive Briefing
        # -------------------------------------------------------------
        _notify("summarize_briefing", "running", "Generating structured executive briefing...")
        state = summarize_briefing_node(state, self.llm)
        if state.errors:
            _notify("summarize_briefing", "error", state.errors[-1])
            return state
        _notify("summarize_briefing", "success", "Executive briefing synthesized successfully")

        # Automatically persist session to disk
        state.save_session(self.config.sessions_dir)
        return state
