"""Identifiable shared state container for the agent graph.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from arxiv_digest.models import (
    ExecutiveBriefing,
    NodeExecutionLog,
    PaperMetadata,
    ParsedPaper,
    QAResponse,
    TextChunk,
)

StepStatus = Literal["success", "warning", "error", "skipped"]


class AgentState(BaseModel):
    """Identifiable shared state persisting across all graph nodes."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_query: str = ""
    intent: Literal["TOPIC_SEARCH", "DIRECT_ID", "UNKNOWN"] = "UNKNOWN"
    parsed_arxiv_id: str | None = None

    # Candidate retrieval & selection
    candidate_papers: list[PaperMetadata] = Field(default_factory=list)
    selected_paper: PaperMetadata | None = None
    selection_rationale: str | None = None

    # PDF parsing & text representation
    pdf_local_path: str | None = None
    parsed_paper: ParsedPaper | None = None

    # Vector store & chunking
    chunks: list[TextChunk] = Field(default_factory=list)
    vector_store_ref: str | None = None

    # Output executive briefing
    briefing: ExecutiveBriefing | None = None

    # QA conversational history & grounded responses
    qa_history: list[dict] = Field(default_factory=list)
    pending_question: str | None = None  # input to the answer_question node

    # Graph traversal: nodes executed so far, in order (across analyze and ask runs)
    visited_nodes: list[str] = Field(default_factory=list)

    # Execution telemetry & resilience tracking
    execution_logs: list[NodeExecutionLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    current_node: str = "initialized"
    is_complete: bool = False

    def log_step(
        self,
        node_name: str,
        status: StepStatus,
        message: str,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an identifiable state transition log."""
        self.current_node = node_name
        self.execution_logs.append(
            NodeExecutionLog(
                node_name=node_name,
                status=status,
                message=message,
                duration_ms=duration_ms,
            )
        )

    def add_error(self, message: str) -> None:
        """Log a failure event without breaking the graph."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Log a non-fatal warning."""
        self.warnings.append(message)

    def record_qa_interaction(self, response: QAResponse) -> None:
        """Append a question-answer turn to the session history."""
        self.qa_history.append(response.model_dump())

    def save_session(self, directory: Path) -> Path:
        """Persist state to disk for future session resumption."""
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"session_{self.session_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return filename

    @classmethod
    def load_session(cls, session_file: Path) -> "AgentState":
        """Restore an existing agent state from disk."""
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
