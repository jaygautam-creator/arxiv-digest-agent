# Autonomous arXiv Paper Digest & QA Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Author:** Jay Gautam (<jaygautam561@gmail.com>)  
> **Target / Organization:** 8byte Engineering Assessment  
> **Repository:** [jaygautam-creator/arxiv-digest-agent](https://github.com/jaygautam-creator/arxiv-digest-agent)

An autonomous, stateful research agent designed to streamline literature review for AI researchers and engineers. Given a natural-language research topic (e.g., *"recent work on KV-cache compression for LLMs"*) or a specific arXiv paper ID/URL, the agent executes an explicit stateful graph to retrieve candidate papers, parse document structure, index chunks into a local vector store, synthesize an executive briefing, and answer follow-up questions grounded in retrieved passages of the paper, refusing when the paper does not contain the answer.

---

## Table of Contents
- [1. Architecture & State Graph](#1-architecture--state-graph)
  - [Graph Flow Diagram](#graph-flow-diagram)
  - [State Shape (`AgentState`)](#state-shape-agentstate)
  - [Graph Nodes & Edge Routing](#graph-nodes--edge-routing)
- [2. Quickstart & Setup](#2-quickstart--setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration & Free-Tier LLMs](#configuration--free-tier-llms)
  - [Free-Tier Rate Limits](#free-tier-rate-limits)
- [3. CLI Usage & Commands](#3-cli-usage--commands)
- [4. Example Run](#4-example-run)
- [5. Retrieval Evaluation](#5-retrieval-evaluation)
- [6. Design Decisions & Tradeoffs](#6-design-decisions--tradeoffs)
- [7. Running Tests](#7-running-tests)
- [8. Project Directory Structure](#8-project-directory-structure)

---

## 1. Architecture & State Graph

### Graph Flow Diagram

```mermaid
flowchart TD
    Start([User Input: Topic or arXiv ID]) --> Q[Node 1: Query Understanding]
    Q --> R[Node 2: arXiv Retrieval]
    
    R --> C1{Intent == TOPIC_SEARCH \n& Candidates > 1?}
    C1 -- Yes --> S[Node 3: Candidate Ranking]
    C1 -- No --> F[Node 4: Fetch & Parse PDF]
    S --> F
    
    F --> CH[Node 5: Section-Aware Chunking]
    CH --> V[Node 6: Local Vector Indexing]
    V --> B[Node 7: Summarize Executive Briefing]
    
    B --> Disk[(Session Persistence to Disk)]
    B --> QA[Interactive Grounded QA Loop]
    
    QA --> QAGate{Query Relevance \n>= Similarity Threshold?}
    QAGate -- Yes --> QAResult[Answer with Section/Page Citations]
    QAGate -- No --> QARefuse[Strict Refusal: Prevents Hallucination]
```

### ASCII Pipeline Representation
```
[User Input] 
      │
      ▼
┌──────────────────────┐
│  Query Understanding │ -> Classifies Intent (DIRECT_ID vs TOPIC_SEARCH)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   arXiv Retrieval    │ -> Queries official arXiv Atom XML feed
└──────────┬───────────┘
           │
           ├─► (Topic Query?) ──► [ Paper Ranking ] ──┐
           │                                          │
           └─► (Direct ID) ───────────────────────────┴─► [ Fetch & Parse PDF ]
                                                                   │
                                                                   ▼
                                                       [ Section-Aware Chunking ]
                                                                   │
                                                                   ▼
                                                       [ Local Vector Indexing ]
                                                                   │
                                                                   ▼
                                                       [ Summarize Briefing ]
                                                                   │
                                                                   ▼
                                                       [ Grounded QA Loop ]
```

### State Shape (`AgentState`)
Identifiable state persists across all nodes and serializes to disk as a complete audit trail:

```python
class AgentState(BaseModel):
    session_id: str                      # Unique 8-character session UUID
    raw_query: str                       # User topic or arXiv ID
    intent: Literal["TOPIC_SEARCH", "DIRECT_ID", "UNKNOWN"]
    parsed_arxiv_id: str | None          # Extracted canonical arXiv ID
    candidate_papers: list[PaperMetadata]# Candidate papers from Atom API
    selected_paper: PaperMetadata | None # Primary chosen paper
    selection_rationale: str | None      # Technical justification for paper selection
    pdf_local_path: str | None           # Cached PDF file path on disk
    parsed_paper: ParsedPaper | None     # Structured sections, abstract, references
    chunks: list[TextChunk]              # Section-bounded chunks with page metadata
    vector_store_ref: str | None         # Local vector store reference
    briefing: ExecutiveBriefing | None   # Structured briefing artifact
    qa_history: list[dict]               # Conversation history for QA turns
    execution_logs: list[NodeExecutionLog]# Millisecond timing audit for each node
    errors: list[str]                    # Handled failure logs
    warnings: list[str]                  # Non-fatal warnings (e.g. OCR fallback)
    is_complete: bool                    # Stage completion flag
```

### Graph Nodes & Edge Routing

1. **`query_understanding`**:
   - Parses regex patterns for modern arXiv IDs (`\d{4}\.\d{4,5}`), legacy identifiers (`cs/0101001`), and arXiv web URLs (`https://arxiv.org/abs/...`).
   - Normalizes input into canonical ID or cleans topic query for boolean retrieval.
2. **`arxiv_retrieval`**:
   - Calls the official arXiv Atom XML feed (`https://export.arxiv.org/api/query`).
   - Filters out conversational words (`"recent"`, `"work"`, `"for"`) to build precise `all:term1 AND all:term2` queries.
   - If nothing matches, relaxes step by step: the two leading terms with AND, then any term with OR.
3. **`paper_ranking`**:
   - The LLM picks the most relevant candidate from titles, dates and abstracts and records a rationale. When the query asks for recent work, the prompt includes today's date and prefers newer papers when relevance is comparable.
   - If the LLM call fails, a keyword-overlap score (plus a recency bonus for "recent" queries) picks the paper instead.
4. **`fetch_and_parse`**:
   - Streams the PDF from arXiv with local disk caching (`data/cache/{arxiv_id}.pdf`).
   - Uses PyMuPDF font information to detect headings: bold lines at body size or larger matching `3`, `3.2 Title`, or a number and title on separate lines, which is how LaTeX renders them. Subsections keep their parent path (`4 Experiments › 4.2 Experimental Results`), and references are split into individual entries.
   - Rebuilds table rows from line positions. LaTeX tables put each cell on its own line, so cells that share a baseline are joined as `row label | v1 | v2 …`, one row per line. Without this, a table came out as one run of numbers, and the LLM attributed values to the wrong rows.
   - Falls back to `pypdf` if PyMuPDF fails, and to the arXiv abstract if the PDF cannot be downloaded or has no text layer (e.g. scanned images).
5. **`chunk_and_embed`**:
   - Chunks never cross a section boundary, and the References section is not indexed.
   - Packs sentences into chunks of about 800 characters with a 150-character overlap. Table rows are kept whole, one per line.
   - Tags each chunk with its section path and its exact page range (e.g. `pp. 21–22`), taken from the PDF block each sentence came from.
6. **`vector_indexing`**:
   - **Hybrid mode** (default when the `embeddings` extra is installed): embeds chunks with `BAAI/bge-small-en-v1.5`, fuses the embedding ranking with a TF-IDF ranking (reciprocal rank fusion), and reranks the top 20 with the `ms-marco-MiniLM-L-6-v2` cross-encoder. Both models run locally on CPU through ONNX, with no PyTorch.
   - **TF-IDF mode** (fallback): sparse TF-IDF vectors (sublinear term frequency, smoothed IDF, stopwords removed) searched by cosine similarity in NumPy.
   - Chunk embeddings are saved in the session file, so resuming a session doesn't re-embed.
7. **`summarize_briefing`**:
   - Builds a context of about 14k characters from the abstract plus the main-body introduction, method, results, limitations and conclusion sections. The budget is shared fairly between sections, and appendices and flattened tables are left out.
   - Validates the LLM's JSON against the `ExecutiveBriefing` schema and re-asks once with the validation error if it is malformed. Title, authors, ID and date always come from arXiv, never from the LLM.
   - If the LLM fails outright, emits an abstract-only briefing whose other fields say "Not available" rather than inventing content.
8. **`qa_loop`**:
   - Interactive CLI REPL (not a graph node; it reads the shared state the graph produced).
   - Retrieves the top 4 chunks. If the question isn't similar enough to any chunk, it refuses without calling the LLM: embedding similarity < 0.55 in hybrid mode, TF-IDF cosine < 0.05 otherwise. Both thresholds were chosen from the evaluation set.
   - Otherwise the LLM answers only from those chunks, citing `[Source n]`. If they don't contain the answer, it must start its reply with `NOT IN PAPER:`, and the answer is shown as not grounded.
   - Every turn is appended to `qa_history`, and the session file is re-saved.

---

## 2. Quickstart & Setup

### Prerequisites
- Python 3.10 or higher
- macOS, Linux, or Windows WSL

### Installation

```bash
git clone https://github.com/jaygautam-creator/arxiv-digest-agent.git
cd arxiv-digest-agent

# with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,embeddings]"

# or with pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,embeddings]"
```

The `embeddings` extra (`fastembed`, ONNX Runtime) enables hybrid retrieval. Its two models (~200 MB) download on first use into `data/models/`. Leave the extra out for a lighter TF-IDF-only install; everything still works, with weaker retrieval (see [Retrieval Evaluation](#5-retrieval-evaluation)). The LLM providers are called over plain HTTPS with `httpx`, so no vendor SDKs are needed.

### Configuration & Free-Tier LLMs
No paid API keys are required. Copy the template and add one free key:

```bash
cp .env.example .env
# then set GROQ_API_KEY=... (recommended) and/or GEMINI_API_KEY=...
```

`.env` is loaded automatically and is git-ignored. With `LLM_PROVIDER=auto` (the default), the agent uses Groq if a Groq key is set, then Gemini, then Ollama (`USE_OLLAMA=true`), and finally the offline mock. Setting `LLM_PROVIDER=gemini` (or `groq`, `ollama`) without its key is a startup error, not a silent switch to mock output.

| Provider | Setup | Default model | Notes |
|---|---|---|---|
| **Groq** (recommended) | `GROQ_API_KEY` from [console.groq.com](https://console.groq.com/) | `openai/gpt-oss-120b` | Most generous free quota; fast |
| **Google Gemini** | `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com/) | `gemini-3.8-flash`, falls back to `gemini-2.5-flash` | Small daily quota (see below) |
| **Local Ollama** | `USE_OLLAMA=true`, `ollama run llama3` | `llama3` | Fully offline; quality depends on the local model |
| **Offline mock** | `--mock` or `MOCK_LLM=true` | – | For tests and CI only. Output is visibly tagged `[MOCK]` placeholder text |

### Free-Tier Rate Limits

A topic run makes **2 LLM calls** (ranking + briefing), plus **1 per QA question**. A direct-ID run skips ranking. The limits below were observed on this project's free accounts in September 2026. Providers change them, and they vary by account.

| Provider | Observed limit | What happens when it is hit |
|---|---|---|
| Groq `openai/gpt-oss-120b` | 1,000 requests/day; **8,000 tokens/minute**. A briefing call is about 5k tokens, so back-to-back runs can hit the per-minute cap | HTTP 429 → the client waits (honoring `Retry-After`) and retries automatically; a run may pause for up to ~20 s |
| Gemini free tier | **20 requests/day per model**, about 6 full runs. Newer Flash models also return HTTP 503 "high demand" at busy times | Gets one short attempt, is then skipped for the rest of the session, and the next model in `GEMINI_FALLBACK_MODELS` is used. The CLI prints which model answered |

If every provider fails, the briefing degrades to arXiv metadata plus the abstract, and QA answers say that the model could not be reached. Neither ever falls back to invented text.

---

## 3. CLI Usage & Commands

```bash
# Analyze by natural-language research topic
python -m arxiv_digest "recent work on KV-cache compression for LLMs"

# Analyze by arXiv ID or URL
python -m arxiv_digest "1706.03762"
python -m arxiv_digest "https://arxiv.org/abs/1706.03762"

# Choose a provider for one run
python -m arxiv_digest "1706.03762" --provider groq

# Export the briefing
python -m arxiv_digest "1706.03762" --export-json briefing.json --export-md briefing.md

# Generate the briefing and exit without entering the QA REPL
python -m arxiv_digest "1706.03762" --no-interactive

# Resume a saved session straight into QA
python -m arxiv_digest --session data/sessions/session_<id>.json

# Offline smoke test without any key (placeholder output)
python -m arxiv_digest "1706.03762" --mock
```

---

## 4. Example Run

Full, unedited transcript with the pipeline trace, all candidates and verification notes: **[`examples/sample_qa_run.md`](examples/sample_qa_run.md)**. The briefing artifacts are [`examples/kv_cache_briefing.md`](examples/kv_cache_briefing.md) and [`.json`](examples/kv_cache_briefing.json).

**Input:** `"recent work on KV-cache compression for LLMs"` (Groq `openai/gpt-oss-120b`, 2026-09-18).

*This run was recorded with TF-IDF retrieval, before hybrid retrieval, table reconstruction and exact page ranges were added. The briefing doesn't depend on retrieval. The QA citations would differ today; for example, the fine-tuning question below is one the evaluation still marks as a miss in both modes.*
arXiv returned 5 candidates (2024–2026). The ranking node selected the most recent directly relevant one, **GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs** ([2605.31105](https://arxiv.org/abs/2605.31105)). The parser found 33 sections, the chunker produced 137 chunks, and the whole run took about 8 s.

**Briefing (excerpt):**

> **Key Results & Claims**
> - On Llama-3.1-8B-Instruct (LongBench, 10% cache budget) GRKV raises the average score from 33.96 to 34.58 with SnapKV and from 36.00 to 36.58 with CriticalKV, improving 14/16 tasks in both cases.
> - On Mistral-7B-Instruct‑v0.3 (LongBench, 10% cache budget) GRKV improves SnapKV from 33.12 to 33.75 and CriticalKV from 33.69 to 34.30, with gains on 12/16 and 14/16 tasks respectively. […]
>
> **Limitations & Edge Cases**
> - Evaluation is limited to three open‑source English models (Llama‑3.1‑8B‑Instruct, Mistral‑7B‑Instruct‑v0.3, Qwen3‑14B) and two long‑context benchmarks (LongBench, RULER); results may not transfer to larger proprietary models or multilingual/multimodal settings.
> - The surrogate prompt‑derived query window is an empirical proxy; misalignment between this window and actual future queries can reduce reconstruction effectiveness […]

I checked the briefing against the PDF. All 20 numbers match with the correct model and benchmark. One method bullet says the objective minimizes "cosine distance", but the paper minimizes squared L2 error. The transcript leaves it uncorrected.

**QA exchanges:**

```
Ask Paper > What does GRKV score on RULER with Llama-3.1-8B-Instruct at a 10% cache budget?
```
> GRKV attains an average RULER score of **29.09** with Llama‑3.1‑8B‑Instruct when the cache budget is limited to 10 % (using the SnapKV eviction method)【Source 2】. (With the alternative CriticalKV eviction method, the score is 41.51【Source 2】.)
>
> *Citations: 4 Experiments › 4.2 Experimental Results (p. 7, score 0.318), plus three appendix chunks. Both numbers match Table 2.*

```
Ask Paper > Does GRKV require fine-tuning the model?
```
> *[Not grounded]* The retrieved passages do not answer this. The provided excerpts do not contain any statement about whether GRKV requires fine‑tuning of the language model. […]
>
> *This is an honest retrieval miss: the paper says "training-free", but lexical retrieval does not link that to "fine-tuning". The model reported the gap instead of guessing.*

```
Ask Paper > What is the capital of France?
```
> *[Refused by similarity gate: best chunk scored 0.0, so no LLM call was made]*
> I cannot answer this question based on the paper. The document does not contain relevant information regarding this query […]

---

## 5. Retrieval Evaluation

`evals/questions.json` holds 33 hand-labelled questions over 5 papers:
- **Lexical** (11): the question reuses the paper's wording.
- **Paraphrase** (16): the same kind of fact, asked in different words.
- **Unanswerable** (2): on-topic, but the paper doesn't say.
- **Out-of-scope** (4): unrelated to the paper.

Each answerable question lists evidence text that must appear in a retrieved chunk, and the runner checks that every evidence string really exists in the parsed paper. It builds the chunks through the agent's own parse → chunk → index nodes.

```bash
python evals/run_eval.py            # retrieval + gate (no LLM calls)
python evals/run_eval.py --llm      # also scores the LLM's answers (~35 calls; mind free-tier quotas)
```

**Retrieval** (final code, top 4 chunks):

| Configuration | Answerable: evidence retrieved | Answerable: wrongly refused | Off-topic refused |
|---|---|---|---|
| TF-IDF, gate 0.15 (original) | 11/27 | 8/27 | 4/4 |
| TF-IDF, gate 0.05 (TF-IDF default now) | 14/27 | 0/27 | 4/4 |
| **Hybrid + reranker (default with the extra)** | **20/27** | **0/27** | **4/4** |

Giving the LLM 6 chunks instead of 4 didn't change the hybrid result (20/27), so top-k stays at 4.

**End-to-end answers** (Groq `gpt-oss-120b`; correct = grounded and contains the expected value):

| Configuration | Answerable answered correctly | Unanswerable + off-topic refused |
|---|---|---|
| TF-IDF, gate 0.05 (final code) | 17/27 | 6/6 |
| Hybrid + reranker (revision before column headers were added) | 18/27 | 6/6 |

**What this says, honestly:**
- Hybrid retrieval clearly finds more evidence (20 vs 14 of 27). On this small set, that shows up as only a one-question gain in final answers.
- The gap is smaller than the retrieval gap for two reasons. The strict evidence strings undercount TF-IDF: it sometimes retrieves a differently worded passage that still answers. And some hybrid losses were answer errors rather than retrieval errors: reading the wrong table column (fixed since by attaching column headers), and one answer given from general knowledge.
- The hybrid end-to-end run on the final code didn't finish, because the eval runs used up the day's free-tier quota. Re-run it with `RETRIEVAL_MODE=hybrid python evals/run_eval.py --llm`.
- With 27 answerable questions, a one- or two-question difference is within noise. Treat these numbers as evidence for the large effects (the gate calibration, 14 → 20 retrieval), not the small ones.

---

## 6. Design Decisions & Tradeoffs

**Explicit graph in plain Python.** Each stage is a function `AgentState → AgentState` in `nodes/`, and `graph.py` wires them in order with one conditional edge (ranking runs only for topic searches with more than one candidate). An error recorded by a node stops the graph at that point. I chose this over LangGraph because a linear pipeline with one branch doesn't need a framework: each node stays unit-testable, and the whole control flow is visible in about 100 lines. The QA loop sits outside the graph because it is interactive. It reads the same state object.

**State and persistence.** `AgentState` is a Pydantic model holding the query, candidates, selected paper, parsed sections, chunks, briefing, QA history and a per-node execution log. It is saved to `data/sessions/session_<id>.json` when the graph finishes and after every QA turn, so `--session` resumes a conversation after a restart. Chunks are saved with their embeddings, so resuming rebuilds the index without re-embedding.

**Retrieval: measured, then chosen.** I started with TF-IDF only: no model downloads, deterministic, sub-millisecond. Before changing it, I built a 33-question evaluation set, and the first finding wasn't about ranking at all. The TF-IDF gate (0.15) refused 8 of 27 answerable questions, because TF-IDF scores drop whenever the wording differs from the paper. After stopword removal, off-topic questions score exactly 0, so recalibrating the gate to 0.05 fixed that without embeddings. Embeddings then earned their place on ranking. Alone, they did *worse* than TF-IDF on these number- and name-heavy papers, but fusing both rankings (RRF) and reranking with a cross-encoder found the answer for 20/27 questions, against 14/27 for TF-IDF. The embedding gate (0.55) also separates off-topic from on-topic questions semantically instead of relying on zero word overlap. The models are an optional extra, so the base install stays light. The TF-IDF side keeps its earlier fixes: stopwords are removed, and hyphenated terms are split so "LLaMA-3-8B" matches "LLaMA-3-Instruct-8B".

**Grounding in layers.** (1) Paper metadata (title, authors, ID, date) is copied from arXiv by code, and the LLM's values for those fields are discarded. (2) A similarity gate refuses out-of-scope questions before any LLM call. (3) The QA prompt allows only the retrieved passages and requires a `NOT IN PAPER:` marker for refusals, which the code checks instead of guessing from wording. (4) Every answer lists the chunks it was given. (5) Tables are rebuilt row by row (`label | v1 | v2`) so values stay attached to their row. The summarizer still leaves tables out: on one run, Groq's model mixed LongBench and RULER scores from table text, and the paper's prose states the headline numbers unambiguously. QA can use tables, and its prompt says to quote a table value only when the row and column are clear.

**Handling the vague and failure cases.**
- **Zero candidates:** relax to the two leading terms, then accept any term.
- **Many candidates:** fetch the top 5 by arXiv relevance and let the ranking node choose, preferring recency for "recent" queries.
- **PDF problems:** download failure or no text layer → the arXiv abstract becomes the only section, with a warning. PyMuPDF failure → `pypdf`. Huge papers are bounded by the 14k-character briefing context and by chunk-level retrieval.
- **LLM problems:** retries with backoff and Gemini model fallback. Malformed JSON → one corrective re-ask. Total failure → an abstract-only briefing that says so.

**What I would do with more time.**
1. Grow the evaluation set. 33 questions over 5 papers is enough to show large effects (like the gate), not small ones, and both gate thresholds were chosen on the same set they are reported on.
2. Attach column headers to table rows. They are often rotated or in a separate block, so rows currently carry only their row label.
3. A second retrieval pass for questions the LLM marks `NOT IN PAPER`, e.g. rewriting the query with the model, since some of those are retrieval misses rather than true absences.
4. OCR (e.g. Tesseract) for scanned PDFs instead of falling back to the abstract.

**Known limitations.**
- Retrieval still misses about a quarter of answerable questions in the eval set, mostly paraphrases whose answer sits in a single sentence ("How large is the implementation?" → "about 3K lines of code"). The agent then says the passages don't answer the question, which is correct behavior but not a useful answer.
- When retrieval misses, the model occasionally answers from general knowledge anyway (in the eval, it described "standard softmax attention" instead of saying the kernel wasn't in the passages). The refusal rules reduce this but don't eliminate it.
- On-topic questions the paper doesn't answer ("GRKV on ImageNet") pass the similarity gate. They are caught by the LLM's `NOT IN PAPER` rule, not by retrieval.
- Hybrid mode adds about 5–30 s per paper for the first embedding pass (CPU) and ~0.5 s per question for reranking.
- Heading detection assumes LaTeX-style bold numbered headings. Unusual templates fall back to fewer, coarser sections. Floating tables can be attributed to the neighbouring section.
- Scanned PDFs without a text layer are summarized from the abstract only. No OCR is used.
- LLM output is not perfect. The example briefing contains one wrong method detail, which is why the briefing is a starting point for reading and not a substitute for it.

---

## 7. Running Tests

44 offline tests use a mock LLM, a fake embedder and a synthetic PDF, so no network, keys or model downloads are needed:

```bash
pytest tests/ -q
# 44 passed in 0.3s
```

They cover query parsing, Atom parsing and query relaxation, heading detection and table-row reconstruction, page-accurate chunking, TF-IDF and hybrid retrieval (gate, fusion, reranking), grounded and refused QA, provider selection, retry and model fallback, briefing validation, the corrective re-ask, and the abstract-only fallback.

---

## 8. Project Directory Structure

```
.
├── README.md
├── pyproject.toml
├── .env.example                 # configuration template (copy to .env)
├── docs/DEVELOPER_GUIDE.md      # conventions for contributors
├── evals/
│   ├── questions.json           # 33 labelled questions over 5 papers
│   └── run_eval.py              # retrieval + end-to-end answer evaluation
├── examples/
│   ├── sample_qa_run.md         # full example run with verification notes
│   ├── kv_cache_briefing.md     # generated briefing (Markdown)
│   └── kv_cache_briefing.json   # generated briefing (JSON)
├── arxiv_digest/
│   ├── cli.py                   # Rich CLI, exports, QA REPL
│   ├── agent.py                 # ArxivDigestAgent Python API
│   ├── graph.py                 # stateful graph orchestration
│   ├── state.py                 # AgentState + session persistence
│   ├── models.py                # Pydantic schemas (ExecutiveBriefing, TextChunk, ...)
│   ├── config.py                # .env loading and provider selection
│   ├── embeddings.py            # optional local embedder + reranker (ONNX)
│   ├── nodes/
│   │   ├── query_parser.py      # intent + arXiv ID/URL parsing
│   │   ├── arxiv_client.py      # Atom API client + query relaxation
│   │   ├── ranker.py            # candidate selection (LLM + recency-aware fallback)
│   │   ├── pdf_parser.py        # download, headings, table rows, references
│   │   ├── chunker.py           # section-bounded chunking
│   │   ├── vector_store.py      # TF-IDF + dense hybrid search, gate, reranking
│   │   ├── summarizer.py        # briefing generation and validation
│   │   └── qa_agent.py          # grounded QA with similarity gate
│   └── llm/
│       ├── base.py              # interface + HTTP retry
│       ├── groq_client.py
│       ├── gemini_client.py     # with model fallback
│       ├── ollama_client.py
│       └── mock_client.py       # offline placeholder provider
└── tests/                       # 44 offline tests
```

---

## License
MIT License. Built by **Jay Gautam** for the **8byte** assessment.
