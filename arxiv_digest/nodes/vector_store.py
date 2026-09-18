"""Node 5 (Part B): Local vector store with lexical and optional dense retrieval.

TF-IDF (sublinear TF, smoothed IDF, stopwords removed) always runs. When an embedder is
available, retrieval becomes hybrid:
  1. Gate: refuse if no chunk's embedding similarity reaches `dense_threshold`.
     On the eval set this separated off-topic from answerable questions cleanly,
     where the TF-IDF gate wrongly refused a third of answerable ones.
  2. Fuse the dense and TF-IDF rankings with reciprocal rank fusion (RRF).
  3. Rerank the top `candidates` with a cross-encoder, if one is configured.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import atexit
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from arxiv_digest.config import AgentConfig
from arxiv_digest.embeddings import Embedder, Reranker, load_retrieval_models
from arxiv_digest.models import TextChunk
from arxiv_digest.state import AgentState

# Function words and question scaffolding. Without this, a question such as
# "What is the capital of France?" matches any chunk on "what/is/the" and slips
# past the similarity gate.
STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be been being before below
    between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just me more most
    my no nor not now of off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    yours tell describe explain paper authors author""".split()
)


RRF_K = 60  # standard reciprocal-rank-fusion constant


def reciprocal_rank_fusion(*score_lists: np.ndarray) -> np.ndarray:
    """Combine rankings by summing 1 / (RRF_K + rank); robust to differently scaled scores."""
    fused = np.zeros(len(score_lists[0]), dtype=np.float64)
    for scores in score_lists:
        for rank, idx in enumerate(np.argsort(-scores)):
            fused[idx] += 1.0 / (RRF_K + rank)
    return fused


class LocalVectorStore:
    """Local vector store: TF-IDF always, plus dense embeddings and reranking when available."""

    def __init__(
        self,
        chunks: list[TextChunk] | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.chunks: list[TextChunk] = chunks or []
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.matrix: np.ndarray | None = None
        self.embedder = embedder
        self.reranker = reranker
        self.dense: np.ndarray | None = None
        if self.chunks:
            self._build_index()
            if embedder is not None:
                self._build_dense_index()

    @property
    def mode(self) -> str:
        if self.dense is None:
            return "tfidf"
        return "hybrid+rerank" if self.reranker is not None else "hybrid"

    def _build_dense_index(self) -> None:
        """Embed chunks, reusing vectors already stored on them (e.g. from a saved session)."""
        stored = [c.dense_embedding for c in self.chunks]
        dims = {len(v) for v in stored if v is not None}
        if any(v is None for v in stored) or len(dims) != 1:
            assert self.embedder is not None  # only called when an embedder was provided
            vectors = self.embedder.embed_documents([c.text for c in self.chunks])
            for chunk, vector in zip(self.chunks, vectors, strict=True):
                chunk.dense_embedding = vector.tolist()
        self.dense = np.array([c.dense_embedding for c in self.chunks], dtype=np.float32)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize into lowercase terms, dropping stopwords.

        Hyphenated compounds are kept whole and also split into their parts, so
        "LLaMA-3-8B" in a question still matches "LLaMA-3-Instruct-8B" in the paper.
        """
        tokens = []
        for token in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower()):
            tokens.append(token)
            if "-" in token:
                tokens.extend(part for part in token.split("-") if len(part) >= 2)
        return [t for t in tokens if t not in STOPWORDS]

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
        self.idf = {term: math.log((num_docs + 1) / (freq + 1)) + 1.0 for term, freq in doc_freqs.items()}

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
        dense_threshold: float = 0.55,
        candidates: int = 20,
    ) -> list[tuple[TextChunk, float]]:
        """Return up to top_k (chunk, score) pairs, or [] when the query fails the relevance gate.

        TF-IDF mode: each chunk must reach `min_threshold`; the score is TF-IDF cosine.
        Hybrid mode: the query must reach `dense_threshold` on its best chunk; the score is
        embedding cosine similarity.
        """
        if self.matrix is None or len(self.chunks) == 0:
            return []

        # Cosine similarity is a dot product because vectors are L2-normalized
        q_vec = self._vectorize_query(query)
        lexical = np.dot(self.matrix, q_vec)

        if self.dense is not None:
            return self._hybrid_search(query, lexical, top_k, dense_threshold, candidates)

        if np.linalg.norm(q_vec) == 0:
            return []
        scores = lexical
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_threshold:
                results.append((self.chunks[idx], score))

        return results

    def _hybrid_search(
        self, query: str, lexical: np.ndarray, top_k: int, dense_threshold: float, candidates: int
    ) -> list[tuple[TextChunk, float]]:
        assert self.dense is not None and self.embedder is not None  # hybrid mode only
        dense = self.dense @ self.embedder.embed_query(query)
        if float(dense.max()) < dense_threshold:
            return []

        pool = np.argsort(-reciprocal_rank_fusion(dense, lexical))[:candidates]
        if self.reranker is not None:
            rerank_scores = self.reranker.score(query, [self.chunks[i].text for i in pool])
            pool = pool[np.argsort(-rerank_scores)]
        return [(self.chunks[i], float(dense[i])) for i in pool[:top_k]]

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
    def load_from_disk(
        cls, file_path: Path, embedder: Embedder | None = None, reranker: Reranker | None = None
    ) -> "LocalVectorStore":
        """Load index from serialized disk artifact."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        chunks = [TextChunk.model_validate(c) for c in data["chunks"]]
        return cls(chunks=chunks, embedder=embedder, reranker=reranker)


def build_vector_store(chunks: list[TextChunk], config: AgentConfig) -> LocalVectorStore:
    """Create a store using the retrieval backend selected in config."""
    embedder, reranker = load_retrieval_models(config)
    return LocalVectorStore(chunks=chunks, embedder=embedder, reranker=reranker)


def retrieve(store: LocalVectorStore, query: str, config: AgentConfig) -> list[tuple[TextChunk, float]]:
    """Search with the thresholds and sizes from config (what the QA node and evals use)."""
    return store.search(
        query,
        top_k=config.retrieval_top_k,
        min_threshold=config.min_similarity_threshold,
        dense_threshold=config.dense_similarity_threshold,
        candidates=config.rerank_candidates,
    )


# In-process registry of built indexes, keyed by session ID
_INDEX_REGISTRY: dict[str, LocalVectorStore] = {}
# Drop stores (and the model sessions they reference) before interpreter teardown.
atexit.register(_INDEX_REGISTRY.clear)


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

    store = build_vector_store(state.chunks, config)
    register_vector_store(state.session_id, store)

    # Persist to disk
    persist_path = config.data_dir / "vector_store" / f"{state.session_id}.json"
    try:
        store.save_to_disk(persist_path)
        state.vector_store_ref = str(persist_path)
    except Exception as e:
        state.add_warning(f"Could not persist vector store to disk: {e}")

    msg = f"Indexed {len(state.chunks)} chunks ({store.mode} retrieval, vocabulary {len(store.vocab)} terms)."
    state.log_step("vector_indexing", "success", msg, (time.time() - start_time) * 1000)

    return state
