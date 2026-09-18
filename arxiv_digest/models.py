"""Data models and schemas for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class PaperMetadata(BaseModel):
    """Normalized metadata for an arXiv paper."""
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    pdf_url: str
    abs_url: str
    published_date: str
    updated_date: str | None = None
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    comment: str | None = None
    doi: str | None = None


class PaperSection(BaseModel):
    """A distinct structural section of an extracted paper."""
    heading: str
    content: str
    page_start: int = 1
    page_end: int = 1
    # Page of each paragraph in `content` (paragraphs are separated by blank lines), when known
    paragraph_pages: list[int] | None = None


class ParsedPaper(BaseModel):
    """Complete parsed output from a downloaded PDF."""
    metadata: PaperMetadata
    sections: list[PaperSection] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    raw_text: str = ""
    parse_warning: str | None = None


class TextChunk(BaseModel):
    """A semantic text chunk with structural provenance for retrieval."""
    chunk_id: str
    section_heading: str
    page_number: int  # page where the chunk's text starts
    text: str
    token_count: int = 0
    dense_embedding: list[float] | None = None
    page_end: int | None = None  # last page the chunk's text reaches, when it spans pages

    @property
    def page_label(self) -> str:
        if self.page_end and self.page_end != self.page_number:
            return f"{self.page_number}–{self.page_end}"
        return str(self.page_number)


class ExecutiveBriefing(BaseModel):
    """Structured executive briefing required by the assessment rubric."""
    title: str
    authors: list[str]
    arxiv_id: str
    publish_date: str
    link: str
    summary_plain_english: str = Field(
        description="One-paragraph plain-English summary highlighting why this paper matters"
    )
    problem_statement: str = Field(
        description="Clear explanation of the core technical bottleneck or research problem addressed"
    )
    method_approach: list[str] = Field(
        description="Key architectural components, algorithms, or methodologies (bullet points)"
    )
    key_results_claims: list[str] = Field(
        description="Quantifiable benchmarks, empirical claims, and comparative findings"
    )
    limitations: list[str] = Field(
        description="Explicit technical constraints, edge cases, or acknowledged gaps"
    )
    suggested_followup_questions: list[str] = Field(
        description="Targeted questions a researcher or practitioner might ask to test or extend the work"
    )

    @field_validator("method_approach", "key_results_claims", "limitations", "suggested_followup_questions")
    @classmethod
    def _drop_blank_items(cls, items: list[str]) -> list[str]:
        return [item.strip() for item in items if item and item.strip()]

    @field_validator("limitations")
    @classmethod
    def _limitations_never_silently_empty(cls, items: list[str]) -> list[str]:
        # The rubric requires explicit limitations; say so rather than render an empty section.
        return items or ["The model reported no limitations; check the paper's own Limitations or Discussion section."]

    def to_markdown(self) -> str:
        """Render the briefing as formatted Markdown."""
        authors_str = ", ".join(self.authors) if self.authors else "Unknown Authors"
        methods = "\n".join(f"- {m}" for m in self.method_approach)
        results = "\n".join(f"- {r}" for r in self.key_results_claims)
        limits = "\n".join(f"- {l}" for l in self.limitations)
        questions = "\n".join(f"- {q}" for q in self.suggested_followup_questions)

        return f"""# Executive Briefing: {self.title}

**Authors:** {authors_str}  
**arXiv ID:** [{self.arxiv_id}]({self.link}) | **Published:** {self.publish_date}

---

## 1. Why This Paper Matters
{self.summary_plain_english}

## 2. Problem Statement
{self.problem_statement}

## 3. Method & Technical Approach
{methods}

## 4. Key Results & Claims
{results}

## 5. Limitations & Edge Cases
{limits}

## 6. Suggested Follow-up Questions
{questions}
"""


class QACitation(BaseModel):
    """Provenance citation linking an answer directly to source text."""
    chunk_id: str
    section: str
    page: int
    excerpt: str
    page_label: str = ""  # e.g. "21–22" when the passage spans pages
    relevance_score: float = 0.0


class QAResponse(BaseModel):
    """Grounded question-answering response with citations and hallucination guard."""
    question: str
    answer: str
    citations: list[QACitation] = Field(default_factory=list)
    is_grounded: bool = True
    confidence_score: float = 1.0


class NodeExecutionLog(BaseModel):
    """Audit log entry for a stage in the stateful graph."""
    node_name: str
    status: Literal["success", "warning", "error", "skipped"]
    message: str
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
