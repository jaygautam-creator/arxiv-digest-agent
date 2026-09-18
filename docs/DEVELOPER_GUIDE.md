# Developer Guide & Project Specification

**Author**: Jay Gautam (<jaygautam561@gmail.com>)  
**Project**: Autonomous arXiv Paper Digest & QA Agent  
**Context**: 8byte Engineering Assessment  

---

## 1. System Overview

The **Autonomous arXiv Paper Digest & QA Agent** is an end-to-end stateful research assistant that automates the scientific literature review workflow for AI engineers and researchers.

Given either:
1. A natural language research topic (e.g., `"recent work on KV-cache compression for LLMs"`), or
2. An exact arXiv ID / URL (e.g., `2401.12345` or `https://arxiv.org/abs/2401.12345`),

the agent executes an explicit stateful graph to:
- Understand intent & formulate targeted arXiv Atom search queries.
- Fetch candidate papers and rank them by relevance to the query.
- Download and parse the paper PDF into structured sections (Abstract, Introduction, Methods, Experiments, Results, Limitations, References).
- Perform section-aware semantic chunking and index chunks into a persistent local vector store.
- Synthesize an Executive Briefing containing: Plain-English summary, Problem statement, Key methods, Claims/results, Limitations, and Follow-up questions.
- Launch an interactive Question-Answering (QA) loop grounded in retrieved chunks, preventing hallucinations and providing section-level citations.

---

## 2. Directory Structure

```
.
├── CLAUDE.md                   # Developer guidance & strict operating rules
├── pyproject.toml              # Modern Python packaging configuration
├── README.md                   # Complete architectural documentation & user guide
├── VIDEO_REFLECTION_SCRIPT.md  # 4-minute video presentation script
├── docs/
│   └── DEVELOPER_GUIDE.md      # This file
├── arxiv_digest/
│   ├── __init__.py             # Public exports
│   ├── config.py               # Config & environment options
│   ├── models.py               # Pydantic schemas (AgentState, PaperMetadata, ExecutiveBriefing)
│   ├── state.py                # State graph container & transition logic
│   ├── graph.py                # Orchestrator coordinating the execution graph
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── query_parser.py     # Intent & arXiv ID parsing
│   │   ├── arxiv_client.py     # Official arXiv Atom feed client
│   │   ├── ranker.py           # Candidate paper ranking & selection
│   │   ├── pdf_parser.py       # PDF downloader & section-aware text extractor
│   │   ├── chunker.py          # Semantic & section-aware text chunker
│   │   ├── vector_store.py     # Local vector store & cosine retrieval
│   │   ├── summarizer.py       # Executive briefing generator
│   │   └── qa_agent.py         # Grounded RAG QA with citation enforcement
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract LLM provider interface
│   │   ├── gemini_client.py    # Google Gemini free tier provider
│   │   ├── groq_client.py      # Groq free tier provider
│   │   ├── ollama_client.py    # Local Ollama provider
│   │   └── mock_client.py      # Deterministic offline mock provider
│   └── cli.py                  # Rich-powered interactive command-line interface
└── tests/
    ├── test_query_parser.py
    ├── test_arxiv_client.py
    ├── test_pdf_parser.py
    ├── test_chunker.py
    ├── test_vector_store.py
    ├── test_graph.py
    └── test_qa_agent.py
```

---

## 3. Engineering Guidelines for Future Tools

1. **Keep Code Human-Authored**: Jay Gautam is the author for 8byte. Never output machine-generated metadata, conversation history, or agent chatter into committed files.
2. **Deterministic & Offline Friendly**: Always ensure unit tests run in offline mode using `MockLLMProvider` so CI/CD and reviewers without API keys can verify the entire pipeline in seconds.
3. **Strict Grounding**: Never allow the QA agent to guess facts not supported by retrieved chunks. If the answer is not present in the paper, it must explicitly state that the paper does not discuss the topic.
