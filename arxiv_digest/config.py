"""Configuration management for the arXiv Digest & QA Agent.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import os
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from a .env file without overriding the real environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def _is_placeholder(value: str | None) -> bool:
    return not value or value.startswith("your_")


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
    gemini_model: str = Field(default="gemini-3.8-flash")
    gemini_fallback_models: list[str] = Field(default_factory=lambda: ["gemini-2.5-flash"])

    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="openai/gpt-oss-120b")
    
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
    min_similarity_threshold: float = Field(default=0.05)

    # Retrieval backend: "tfidf" (lexical only), "hybrid" (TF-IDF + local embeddings + reranker),
    # or "auto" (hybrid when the optional fastembed extra is installed). The model default is tfidf
    # so tests never download models; from_env() defaults to auto.
    retrieval_mode: str = Field(default="tfidf")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    reranker_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
    dense_similarity_threshold: float = Field(default=0.55)
    rerank_candidates: int = Field(default=20)
    request_timeout: float = Field(default=30.0)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from `.env` and environment variables.

        `LLM_PROVIDER` selects the provider explicitly; when unset (or "auto"), the first
        provider with a real API key wins, and mock is used only if none is configured.
        Groq is preferred over Gemini because its free tier allows far more requests per day.
        """
        load_dotenv()
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = None if _is_placeholder(gemini_key) else gemini_key
        groq_key = None if _is_placeholder(groq_key) else groq_key

        requested = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        if os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes"):
            provider = LLMProviderType.MOCK
        elif requested in {p.value for p in LLMProviderType}:
            provider = LLMProviderType(requested)
        elif groq_key:
            provider = LLMProviderType.GROQ
        elif gemini_key:
            provider = LLMProviderType.GEMINI
        elif os.getenv("USE_OLLAMA", "").lower() in ("1", "true"):
            provider = LLMProviderType.OLLAMA
        else:
            provider = LLMProviderType.MOCK

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
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            gemini_fallback_models=[
                m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash").split(",") if m.strip()
            ],
            groq_api_key=groq_key,
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
            data_dir=data_dir,
            cache_dir=cache_dir,
            sessions_dir=sessions_dir,
            arxiv_max_results=int(os.getenv("ARXIV_MAX_RESULTS", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
            min_similarity_threshold=float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.05")),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "auto").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            reranker_model=os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
            dense_similarity_threshold=float(os.getenv("DENSE_SIMILARITY_THRESHOLD", "0.55")),
            rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
        )
