"""High-level Python API for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from pathlib import Path

from arxiv_digest.config import AgentConfig
from arxiv_digest.graph import ProgressCallback, ResearchStateGraph
from arxiv_digest.llm import get_llm_provider
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.models import QAResponse
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
        """Answer one question by running the graph's `ask` entry point (answer_question → persist_session).

        The turn is appended to `state.qa_history` and the session file is re-saved.
        """
        state = self.graph.ask(state, question)
        if state.current_node == "error":
            return QAResponse(question=question, answer=state.errors[-1], is_grounded=False, confidence_score=0.0)
        return QAResponse.model_validate(state.qa_history[-1])

    def load_session(self, session_path: str | Path) -> AgentState:
        """Restore a previously analyzed paper session from disk."""
        return AgentState.load_session(Path(session_path))
