"""Tests for local vector database & similarity search.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from pathlib import Path
from arxiv_digest.models import TextChunk
from arxiv_digest.nodes.vector_store import LocalVectorStore


def test_vector_store_search_and_persistence(tmp_path: Path):
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            section_heading="KV Cache",
            page_number=2,
            text="KV-cache compression techniques significantly decrease GPU memory consumption during decoding.",
        ),
        TextChunk(
            chunk_id="chunk_2",
            section_heading="Quantization",
            page_number=4,
            text="4-bit weight quantization allows running massive LLMs on standard consumer edge devices.",
        ),
        TextChunk(
            chunk_id="chunk_3",
            section_heading="Robotics",
            page_number=7,
            text="Quadruped robots employ reinforcement learning controllers for rugged terrain navigation.",
        ),
    ]

    store = LocalVectorStore(chunks=chunks)
    assert len(store.vocab) > 0

    # Search for KV cache
    results = store.search("KV-cache memory consumption", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_1"
    assert score > 0.3

    # Persistence roundtrip
    saved_file = tmp_path / "test_store.json"
    store.save_to_disk(saved_file)
    loaded_store = LocalVectorStore.load_from_disk(saved_file)

    loaded_results = loaded_store.search("KV-cache memory", top_k=1)
    assert loaded_results[0][0].chunk_id == "chunk_1"
