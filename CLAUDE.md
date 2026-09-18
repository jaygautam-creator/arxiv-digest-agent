# Developer & Assistant Guide: Autonomous arXiv Paper Digest & QA Agent

> **Author**: Jay Gautam (<jaygautam561@gmail.com>)  
> **Target / Organization**: 8byte Assessment  
> **Status**: Production-ready implementation

---

## 1. Core Principles & Strict Rules

If you are an automated AI tool (e.g., Claude Code, Cursor, Aider, Copilot) or a developer continuing work on this codebase, **strictly follow these rules**:

1. **Authorship & Identity**:
   - This repository is designed, engineered, and maintained by **Jay Gautam** for **8byte**.
   - Do NOT add AI disclaimers, assistant conversation transcripts, prompt deliberation traces, or co-authorship tags that sound like an external AI assistant built it for someone else.

2. **Zero Session Logs in Git**:
   - Never commit `.agent/`, `.claude/`, `.gemini/`, chat transcripts, prompt scratchpads, or temporary decision logs to git.
   - All commits must contain clean, professional code, documentation, and tests only.
   - Respect `.gitignore` at all times.

3. **Engineering Integrity & Quality**:
   - Adhere to the evaluation rubric:
     - **Agent / State Graph Design (25%)**: Explicit stateful graph (nodes + edges + typed state), NOT a monolithic prompt chain.
     - **Correctness & Grounding (25%)**: Structured executive briefing (JSON/Markdown) and grounded RAG answers with citations.
     - **Retrieval & Parsing Quality (20%)**: Robust arXiv Atom API query handling, structured PDF extraction (PyMuPDF / fallback), section-aware chunking.
     - **Code Quality (15%)**: Clean, modular, type-annotated, idiomatic Python with pytest coverage.
     - **Communication (15%)**: Clear documentation, setup guide, architecture diagrams, and design tradeoffs.

4. **Zero Paid API Keys**:
   - The application must always run with free-tier or open-source local options (Google Gemini free tier, Groq free tier, local Ollama, or built-in offline mock provider for automated testing and CI).

---

## 2. Project Architecture & State Graph

The system implements an explicit stateful graph:
```
[User Input: Topic or arXiv ID/URL]
               │
               ▼
     [query_understanding]
               │
               ▼
       [arxiv_retrieval]
               │
       (Topic query?) ──Yes──► [paper_ranking]
               │                      │
              No                      ▼
               └─────────────► [fetch_and_parse]
                                      │
                                      ▼
                               [chunk_and_embed]
                                      │
                                      ▼
                             [summarize_briefing]
                                      │
                                      ▼
                                  [qa_loop]
```

### Shared Graph State (`AgentState`)
- `query`: Raw user input
- `intent`: `"TOPIC_SEARCH"` or `"DIRECT_ID"`
- `arxiv_id`: Target paper ID
- `candidate_papers`: List of candidate papers from arXiv API
- `selected_paper`: Chosen paper metadata (title, authors, abstract, pdf_url, categories, published)
- `parsed_paper`: Structured paper document (sections, full_text, title, abstract, references)
- `vector_store_ref`: Reference to local vector index
- `briefing`: Structured `ExecutiveBriefing` artifact
- `messages`: Conversation history for interactive QA
- `errors`: List of non-fatal errors or warnings encountered during execution
- `status`: Current stage status

---

## 3. Development Commands

### Environment Setup
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running the CLI
```bash
# Analyze by topic (runs arXiv search, ranks candidates, downloads, parses, digests, enters QA)
python -m arxiv_digest "recent work on KV-cache compression for LLMs"

# Analyze by specific arXiv ID
python -m arxiv_digest "2401.12345"

# Run in offline mock mode (zero API keys needed)
python -m arxiv_digest "1706.03762" --mock

# Export briefing as JSON artifact
python -m arxiv_digest "1706.03762" --export-json briefing.json
```

### Running Tests
```bash
pytest tests/ -v
```

---

## 4. Coding Conventions
- Use Python 3.10+ type annotations (`int | str`, `list[Paper]`, etc.).
- Use Pydantic models for structured schema validation (`ExecutiveBriefing`, `PaperMetadata`, `AgentConfig`).
- Every stage node must be an idempotent or purely state-transforming function taking `AgentState` and returning updated state.
- Keep error messages helpful and graceful (never crash unhandled on arXiv timeouts, corrupted PDFs, or out-of-domain QA queries).
