"""Tests for hybrid retrieval (dense gate, RRF fusion, reranking) using a fake embedder.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from unittest.mock import patch

import numpy as np

from arxiv_digest.config import AgentConfig
from arxiv_digest.embeddings import load_retrieval_models
from arxiv_digest.models import TextChunk
from arxiv_digest.nodes.vector_store import LocalVectorStore, reciprocal_rank_fusion

VOCAB = ["gpu", "hardware", "p100", "trained", "smoothing", "label", "regularization", "cake"]
SYNONYMS = {"hardware": "gpu", "regularization": "smoothing"}


class FakeEmbedder:
    """Bag-of-words over a tiny vocabulary, with synonyms mapped together to mimic semantics."""

    def __init__(self):
        self.document_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(VOCAB), dtype=np.float32)
        for word in text.lower().replace("?", "").split():
            word = SYNONYMS.get(word, word)
            if word in VOCAB:
                v[VOCAB.index(word)] += 1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        self.document_calls += 1
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class ReverseReranker:
    """Prefers whichever candidate comes last, to prove the reranker controls the final order."""

    def score(self, query, texts):
        return np.arange(len(texts), dtype=np.float32)


def _chunks():
    texts = [
        "We trained on 8 P100 gpu machines.",
        "We used label smoothing during training.",
        "Unrelated appendix text about tables.",
    ]
    return [TextChunk(chunk_id=f"c{i}", section_heading="S", page_number=1, text=t) for i, t in enumerate(texts)]


def test_dense_similarity_finds_paraphrases_tfidf_misses():
    chunks = _chunks()
    lexical_only = LocalVectorStore(chunks=chunks)
    hybrid = LocalVectorStore(chunks=_chunks(), embedder=FakeEmbedder())

    assert lexical_only.search("What hardware?", min_threshold=0.1) == []
    results = hybrid.search("What hardware?", top_k=1, dense_threshold=0.5)
    assert results and results[0][0].chunk_id == "c0"


def test_dense_gate_refuses_off_topic_questions():
    hybrid = LocalVectorStore(chunks=_chunks(), embedder=FakeEmbedder())
    assert hybrid.search("How do I bake a cake?", dense_threshold=0.5) == []


def test_reranker_decides_final_order():
    query = "label smoothing regularization"
    plain = LocalVectorStore(chunks=_chunks(), embedder=FakeEmbedder())
    reranked = LocalVectorStore(chunks=_chunks(), embedder=FakeEmbedder(), reranker=ReverseReranker())

    fused_ids = [c.chunk_id for c, _ in plain.search(query, top_k=3, dense_threshold=0.1)]
    reranked_ids = [c.chunk_id for c, _ in reranked.search(query, top_k=3, dense_threshold=0.1)]
    assert reranked_ids == list(reversed(fused_ids))


def test_stored_embeddings_are_reused():
    chunks = _chunks()
    embedder = FakeEmbedder()
    LocalVectorStore(chunks=chunks, embedder=embedder)
    LocalVectorStore(chunks=chunks, embedder=embedder)  # e.g. a resumed session
    assert embedder.document_calls == 1
    assert all(c.dense_embedding is not None for c in chunks)


def test_rrf_rewards_agreement_between_rankings():
    a = np.array([0.9, 0.5, 0.1])
    b = np.array([0.8, 0.1, 0.6])
    assert int(np.argmax(reciprocal_rank_fusion(a, b))) == 0


def test_tfidf_mode_and_missing_extra_load_no_models():
    assert load_retrieval_models(AgentConfig(retrieval_mode="tfidf")) == (None, None)
    with patch("arxiv_digest.embeddings.fastembed_available", return_value=False):
        assert load_retrieval_models(AgentConfig(retrieval_mode="auto")) == (None, None)
