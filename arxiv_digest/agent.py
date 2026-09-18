"""High-level Python API for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from pathlib import Path
from arxiv_digest.config import AgentConfig
from arxiv_digest.graph import ProgressCallback, ResearchStateGraph
from arxiv_digest.llm import get_llm_provider
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import ExecutiveBriefing, QAResponse
from arxiv_digest.nodes.qa_agent import answer_question
from arxiv_digest.state import AgentState


class ArxivDigestAgent:
    """End-to-end autonomous agent for arXiv paper digestion and grounded QA."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: BaseLLM | None = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.llm = llm or get_llm_provider(self.config)
        self.graph = ResearchStateGraph(config=self.config, llm=self.llm)

    def analyze(
        self,
        query: str,
        on_progress: ProgressCallback | None = None,
    ) -> AgentState:
        """Run the state graph for a topic query or direct arXiv ID."""
        return self.graph.execute(query=query, on_progress=on_progress)

    def ask(self, state: AgentState, question: str) -> QAResponse:
        """Query the paper using grounded retrieval-augmented generation."""
        return answer_question(
            question=question,
            state=state,
            llm=self.llm,
            config=self.config,
        )

    def load_session(self, session_path: str | Path) -> AgentState:
        """Restore a previously analyzed paper session from disk."""
        return AgentState.load_session(Path(session_path))
