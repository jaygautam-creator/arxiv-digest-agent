"""Autonomous arXiv Paper Digest & QA Agent.

An explicit stateful agent that fetches arXiv papers, extracts structured
sections, synthesizes executive briefings, and powers grounded question answering.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.agent import ArxivDigestAgent
from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.models import (
    ExecutiveBriefing,
    PaperMetadata,
    PaperSection,
    ParsedPaper,
    QACitation,
    QAResponse,
    TextChunk,
)
from arxiv_digest.nodes.qa_agent import answer_question
from arxiv_digest.state import AgentState

__version__ = "0.1.0"
__author__ = "Jay Gautam <jaygautam561@gmail.com>"

__all__ = [
    "ArxivDigestAgent",
    "AgentState",
    "AgentConfig",
    "LLMProviderType",
    "ExecutiveBriefing",
    "PaperMetadata",
    "PaperSection",
    "ParsedPaper",
    "TextChunk",
    "QACitation",
    "QAResponse",
    "answer_question",
]
