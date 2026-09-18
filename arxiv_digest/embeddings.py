"""Optional local dense retrieval models (embedder + cross-encoder reranker).

Both run on CPU through fastembed/ONNX, so there is no PyTorch dependency. They are an
optional extra (`pip install -e ".[embeddings]"`); without it, retrieval falls back to
TF-IDF alone. Models download once (~200 MB total) into `<data_dir>/models`.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import atexit
import gc
import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

from arxiv_digest.config import AgentConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized vector per text, shape (n, dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return one L2-normalized query vector, shape (dim,)."""


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Return a relevance score per text (higher is more relevant)."""


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.query_embed([text])), dtype=np.float32))[0]


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.rerank(query, texts)), dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=4)
def _load(embedding_model: str, reranker_model: str, cache_dir: str) -> tuple[Embedder, Reranker | None]:
    embedder = FastEmbedEmbedder(embedding_model, cache_dir)
    reranker = FastEmbedReranker(reranker_model, cache_dir) if reranker_model else None
    return embedder, reranker


@atexit.register
def _release_models() -> None:
    # ONNX Runtime sessions that survive into interpreter teardown can abort the process
    # on macOS ("recursive_mutex lock failed"); free them while Python is still intact.
    _load.cache_clear()
    gc.collect()


def load_retrieval_models(config: AgentConfig) -> tuple[Embedder | None, Reranker | None]:
    """Return (embedder, reranker) for the configured retrieval mode, or (None, None) for TF-IDF only.

    "auto" uses dense retrieval when fastembed is installed; "hybrid" requires it; "tfidf" never uses it.
    Models are loaded once per process.
    """
    mode = config.retrieval_mode
    if mode == "tfidf" or (mode == "auto" and not fastembed_available()):
        return None, None
    if not fastembed_available():
        raise ImportError('RETRIEVAL_MODE=hybrid needs the embeddings extra: pip install -e ".[embeddings]"')
    try:
        return _load(config.embedding_model, config.reranker_model, str(config.data_dir / "models"))
    except Exception as e:
        if mode == "hybrid":
            raise
        logger.warning("Could not load embedding models (%s); using TF-IDF retrieval only.", e)
        return None, None
