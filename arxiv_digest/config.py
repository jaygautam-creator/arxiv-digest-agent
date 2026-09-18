"""Configuration management for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import os
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field


class LLMProviderType(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"
    MOCK = "mock"


class AgentConfig(BaseModel):
    """Configuration settings for the agent pipeline."""

    # Provider & Model selection
    provider: LLMProviderType = Field(default=LLMProviderType.MOCK)
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")

    # Storage paths
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./data/cache"))
    sessions_dir: Path = Field(default=Path("./data/sessions"))

    # Processing & Retrieval parameters
    arxiv_max_results: int = Field(default=5)
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    retrieval_top_k: int = Field(default=4)
    min_similarity_threshold: float = Field(default=0.15)
    request_timeout: float = Field(default=30.0)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables with graceful fallback."""
        # Auto-detect available providers
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        force_mock = os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes")

        provider = LLMProviderType.MOCK
        if force_mock:
            provider = LLMProviderType.MOCK
        elif gemini_key:
            provider = LLMProviderType.GEMINI
        elif groq_key:
            provider = LLMProviderType.GROQ
        elif os.getenv("USE_OLLAMA", "").lower() in ("1", "true"):
            provider = LLMProviderType.OLLAMA

        data_dir = Path(os.getenv("AGENT_DATA_DIR", "./data"))
        cache_dir = data_dir / "cache"
        sessions_dir = data_dir / "sessions"

        # Ensure directories exist
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            provider=provider,
            gemini_api_key=gemini_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            groq_api_key=groq_key,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
            data_dir=data_dir,
            cache_dir=cache_dir,
            sessions_dir=sessions_dir,
            arxiv_max_results=int(os.getenv("ARXIV_MAX_RESULTS", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
            min_similarity_threshold=float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.15")),
        )
