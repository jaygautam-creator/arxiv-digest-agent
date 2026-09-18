"""Node 5 (Part B): Local Vector Database & Hybrid Retrieval Engine.

Implements a robust, self-contained local vector index using normalized TF-IDF / sublinear
token representations and cosine distance calculation with persistent disk serialization.
Ensures zero external cloud vector DB dependency and runs instantly on any machine.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np

from arxiv_digest.config import AgentConfig
from arxiv_digest.models import TextChunk
from arxiv_digest.state import AgentState


class LocalVectorStore:
    """Local, lightweight vector store with TF-IDF cosine similarity search."""

    def __init__(self, chunks: list[TextChunk] | None = None):
        self.chunks: list[TextChunk] = chunks or []
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.matrix: np.ndarray | None = None
        if self.chunks:
            self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def _build_index(self) -> None:
        """Build term-document matrix with sublinear TF and smoothed IDF."""
        num_docs = len(self.chunks)
        if num_docs == 0:
            return

        doc_tokens = [self._tokenize(c.text) for c in self.chunks]
        doc_freqs: Counter[str] = Counter()

        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] += 1

        # Build vocabulary
        self.vocab = {term: idx for idx, (term, freq) in enumerate(doc_freqs.items())}
        num_terms = len(self.vocab)

        # Compute smoothed IDF: log((N + 1) / (df + 1)) + 1
        self.idf = {
            term: math.log((num_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freqs.items()
        }

        # Build TF-IDF document matrix
        self.matrix = np.zeros((num_docs, num_terms), dtype=np.float32)
        for doc_idx, tokens in enumerate(doc_tokens):
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.vocab:
                    term_idx = self.vocab[term]
                    # Sublinear TF scaling: 1 + log(tf)
                    tf = 1.0 + math.log(count)
                    self.matrix[doc_idx, term_idx] = tf * self.idf[term]

            # L2 normalize document vector
            norm = np.linalg.norm(self.matrix[doc_idx])
            if norm > 0:
                self.matrix[doc_idx] /= norm

    def _vectorize_query(self, query: str) -> np.ndarray:
        """Project query string into indexed vector space."""
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(tokens)
        for term, count in counts.items():
            if term in self.vocab:
                term_idx = self.vocab[term]
                tf = 1.0 + math.log(count)
                vec[term_idx] = tf * self.idf[term]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_threshold: float = 0.05,
    ) -> list[tuple[TextChunk, float]]:
        """Compute cosine similarity between query vector and document index."""
        if self.matrix is None or len(self.chunks) == 0:
            return []

        q_vec = self._vectorize_query(query)
        if np.linalg.norm(q_vec) == 0:
            return []

        # Cosine similarity is dot product because vectors are L2-normalized
        scores = np.dot(self.matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_threshold:
                results.append((self.chunks[idx], score))

        return results

    def save_to_disk(self, file_path: Path) -> None:
        """Serialize chunks and index metadata to local JSON."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self.chunks],
            "vocab": self.vocab,
            "idf": self.idf,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_from_disk(cls, file_path: Path) -> "LocalVectorStore":
        """Load index from serialized disk artifact."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [TextChunk.model_validate(c) for c in data["chunks"]]
        instance = cls(chunks=chunks)
        return instance


# Global or session index registry
_INDEX_REGISTRY: dict[str, LocalVectorStore] = {}


def get_vector_store(session_id: str) -> LocalVectorStore | None:
    """Retrieve vector store for an active session."""
    return _INDEX_REGISTRY.get(session_id)


def register_vector_store(session_id: str, store: LocalVectorStore) -> None:
    """Register vector store in active memory."""
    _INDEX_REGISTRY[session_id] = store


def index_chunks_node(state: AgentState, config: AgentConfig) -> AgentState:
    """Graph Node: Build and persist local vector index for the chunked paper."""
    start_time = time.time()

    if not state.chunks:
        msg = "No chunks available to index in vector store."
        state.add_error(msg)
        state.log_step("vector_indexing", "error", msg, (time.time() - start_time) * 1000)
        return state

    store = LocalVectorStore(chunks=state.chunks)
    register_vector_store(state.session_id, store)

    # Persist to disk
    persist_path = config.data_dir / "vector_store" / f"{state.session_id}.json"
    try:
        store.save_to_disk(persist_path)
        state.vector_store_ref = str(persist_path)
    except Exception as e:
        state.add_warning(f"Could not persist vector store to disk: {e}")

    msg = f"Indexed {len(state.chunks)} chunks into LocalVectorStore (Vocabulary: {len(store.vocab)} terms)."
    state.log_step("vector_indexing", "success", msg, (time.time() - start_time) * 1000)

    return state
