# Developer Guide

**Author:** Jay Gautam (<jaygautam561@gmail.com>) · **Project:** Autonomous arXiv Paper Digest & QA Agent (8byte assessment)

This guide is for working on the code. For usage, architecture and design tradeoffs, see the [README](../README.md).

---

## 1. Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # add GROQ_API_KEY and/or GEMINI_API_KEY for live runs
pytest tests/ -q            # offline; no keys or network needed
```

## 2. Code map

| Module | Responsibility |
|---|---|
| `graph.py` | Runs the nodes in order; the one conditional edge (ranking only for topic searches with >1 candidate); stops at the first recorded error |
| `state.py` | `AgentState` (shared state) and JSON session save/load |
| `models.py` | Pydantic schemas: `PaperMetadata`, `PaperSection`, `TextChunk`, `ExecutiveBriefing`, `QAResponse` |
| `config.py` | `.env` loading and provider selection (`LLM_PROVIDER`, keys, models, retrieval parameters) |
| `nodes/*.py` | One module per graph stage; each exposes a `*_node(state, ...) -> AgentState` function |
| `llm/base.py` | `BaseLLM` interface and `post_with_retry` (backoff on 429/5xx, honours `Retry-After`) |
| `llm/*_client.py` | Groq, Gemini (with model fallback), Ollama, and an offline mock |
| `agent.py` / `cli.py` | Python API and the Rich CLI / QA REPL |

## 3. Conventions

- **Nodes are state transformers.** Take `AgentState`, return it. Record outcomes with `state.log_step(...)`,
  recoverable problems with `state.add_warning(...)`, and fatal ones with `state.add_error(...)`, which stops the graph.
  Don't raise out of a node.
- **Never let the LLM supply facts that code already knows.** Paper metadata comes from arXiv, and briefing
  fields that could not be generated say "Not available" rather than holding filler text.
- **QA answers must stay grounded.** Anything that changes retrieval or prompts should keep
  `tests/test_qa_agent.py` and the stopword-gate test in `tests/test_vector_store.py` passing.
- **Providers fail loudly.** A selected provider without its key raises `ProviderConfigError`; nothing silently
  switches to the mock. Mock output is tagged `[MOCK]`.
- Python 3.10+ type hints (`list[str]`, `X | None`), Pydantic v2 models, and `logging` rather than `print` outside the CLI.

## 4. Common changes

**Adding an LLM provider:** subclass `BaseLLM` in `llm/<name>_client.py`, implement `generate(prompt, system_prompt,
json_mode)` using `post_with_retry`, add a value to `LLMProviderType`, and wire it in `llm/__init__.py:get_llm_provider`
and `describe_provider`.

**Adding a graph node:** write `nodes/<name>.py` with a `<name>_node(state, ...)` function, add any new fields to
`AgentState`, and insert the call (with its progress notification) in `graph.py`.

**Tuning retrieval:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K` and `MIN_SIMILARITY_THRESHOLD` are read from `.env`.
Check changes against a real paper as well as the unit tests. The out-of-scope gate is sensitive to the threshold.

## 5. Testing

All tests run offline. LLM behaviour is simulated with `MockLLM` or small `BaseLLM` stubs, HTTP with
`unittest.mock.patch`, and PDFs are generated on the fly with PyMuPDF (`tests/test_pdf_parser.py`).
Before changing prompts or parsing, do a live run and check the briefing against the paper. Unit tests
cannot catch a plausible but wrong summary.
