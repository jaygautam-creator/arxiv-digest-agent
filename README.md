# Autonomous arXiv Paper Digest & QA Agent

[![CI](https://github.com/jaygautam-creator/arxiv-digest-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jaygautam-creator/arxiv-digest-agent/actions/workflows/ci.yml)
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
  - [What Each Node Does](#what-each-node-does)
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

The graph is defined as data in [`arxiv_digest/graph.py`](arxiv_digest/graph.py): named nodes, edges with routing conditions, and two entry points. One runner executes it. The diagram below is generated from that definition (`python -m arxiv_digest --graph`), and a test fails if it drifts from the code.

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}}}%%
flowchart TD
    analyze_in([analyze]) --> query_understanding
    ask_in([ask]) --> answer_question
    subgraph analyze_flow [analyze]
        query_understanding["`query_understanding
classify topic vs arXiv ID`"]
        arxiv_retrieval["`arxiv_retrieval
query the arXiv Atom API`"]
        paper_ranking["`paper_ranking
choose one candidate`"]
        fetch_and_parse["`fetch_and_parse
download PDF, extract sections`"]
        chunk_and_embed["`chunk_and_embed
section-bounded chunks`"]
        vector_indexing["`vector_indexing
TF-IDF / hybrid index`"]
        summarize_briefing["`summarize_briefing
structured briefing`"]
    end
    subgraph ask_flow [ask]
        answer_question["`answer_question
grounded RAG answer`"]
    end
    persist_session["`persist_session
save state as JSON`"]
    query_understanding --> arxiv_retrieval
    arxiv_retrieval -- topic search --> paper_ranking
    arxiv_retrieval -- direct ID --> fetch_and_parse
    paper_ranking --> fetch_and_parse
    fetch_and_parse --> chunk_and_embed
    chunk_and_embed --> vector_indexing
    vector_indexing --> summarize_briefing
    summarize_briefing --> persist_session
    answer_question --> persist_session
    persist_session --> finished([end])
    failed([error: stop, errors kept in state])
    analyze_flow -. any node error .-> failed
    ask_flow -. any node error .-> failed
    persist_session -. error .-> failed
```

- **Entry points.** `analyze` runs retrieval through the briefing. `ask` runs one QA turn on the same state. The interactive REPL just calls `ask` once per question.
- **Conditional edge.** After `arxiv_retrieval`, topic searches go to `paper_ranking`, while direct IDs (already resolved to one paper) skip straight to `fetch_and_parse`. Edges are checked in declaration order, and every node must have an unconditional fallback edge (`validate()` enforces this).
- **Error routing.** A node reports failure by adding to `state.errors`. The runner then stops at the `error` terminal and returns the state with everything gathered so far. Recoverable problems (a PDF that won't download, a failed LLM summary) are handled inside the node and recorded as warnings, so the graph continues.
- **Persistence** is its own node, shared by both entry points, so the session file is written after every analysis and every QA turn.

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
    qa_history: list[dict]               # One entry per QA turn: question, answer, citations
    pending_question: str | None         # Input to the answer_question node
    visited_nodes: list[str]             # Path taken through the graph, in order
    current_node: str                    # Last node, or "end" / "error"
    execution_logs: list[NodeExecutionLog]# Per-node status, message and duration
    errors: list[str]                    # Failures that stopped the graph
    warnings: list[str]                  # Recoverable problems (e.g. abstract-only fallback)
    is_complete: bool                    # Briefing produced
```

### What Each Node Does

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
   - One extra chunk holds the paper's arXiv metadata (title, authors, date, ID, categories), so questions like "who are the authors?" are answered and cited from the arXiv record.
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
8. **`answer_question`** (entry point `ask`, run once per question):
   - Retrieves the top 4 chunks. It refuses without calling the LLM only when no signal finds the question relevant. In hybrid mode that means embedding similarity < 0.55 **and** TF-IDF < 0.05, so off-topic questions fail both, while short factual ones ("who is author") pass on keywords. In TF-IDF mode it means TF-IDF < 0.05. Thresholds were chosen from the evaluation set.
   - Otherwise the LLM answers only from those chunks, citing `[Source n]`. If they don't contain the answer, it must start its reply with `NOT IN PAPER:`, and the answer is shown as not grounded.
   - Every turn is appended to `qa_history`.
9. **`persist_session`**: writes the whole state to `data/sessions/session_<id>.json`, after both `analyze` and every `ask`.

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

Full, unedited transcript with the graph trace, all candidates, every QA citation and verification notes: **[`examples/sample_qa_run.md`](examples/sample_qa_run.md)**. The briefing artifacts are [`examples/kv_cache_briefing.md`](examples/kv_cache_briefing.md) and [`.json`](examples/kv_cache_briefing.json).

**Input:** `"recent work on KV-cache compression for LLMs"` (Groq `openai/gpt-oss-120b`, hybrid retrieval, 2026-09-19).

arXiv returned 5 candidates (2024–2026). The ranking node selected the most recent directly relevant one, **GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs** ([2605.31105](https://arxiv.org/abs/2605.31105)). The parser found 33 sections, and the chunker produced 159 chunks. Path: `query_understanding → arxiv_retrieval → paper_ranking → fetch_and_parse → chunk_and_embed → vector_indexing → summarize_briefing → persist_session`.

**Briefing (excerpt):**

> **Key Results & Claims**
> - On Llama-3.1-8B‑Instruct with a 10% cache budget, GRKV raises the LongBench average score from 33.96 to 34.58 when paired with SnapKV and from 36.00 to 36.58 when paired with CriticalKV (improving 14/16 tasks in both cases)
> - On Mistral-7B‑Instructv0.3 with a 10% cache budget, GRKV improves SnapKV’s LongBench average from 33.12 to 33.75 (12/16 tasks) and CriticalKV’s from 33.69 to 34.30 (14/16 tasks)
> - […three ablation results…]
>
> **Limitations & Edge Cases**
> - Evaluations are limited to three open‑source models (Llama‑3.1‑8B‑Instruct, Mistral‑7B‑Instructv0.3, Qwen3‑14B) and two English‑centric long‑context benchmarks (LongBench and RULER); results may not generalize to larger proprietary models, multilingual or multimodal settings
> - […]

I checked the briefing against the PDF. All 24 numbers appear in the paper and are attributed to the right model and benchmark. One overstatement: the summary says GRKV "eliminates" over-merging, where the paper says "mitigates". The transcript leaves it unedited.

**QA exchanges** (each one runs the graph's `ask` entry point):

```
Ask Paper > What does GRKV score on RULER with Llama-3.1-8B-Instruct at a 10% cache budget?
```
> GRKV improves the RULER scores for Llama‑3.1‑8B‑Instruct under a 10 % cache budget:
> 
> * **SnapKV + GRKV:** average score = **29.09**【Source 2】 (also noted in the ablation table as the best average when λₖ = λᵥ = 10⁻²【Source 1】).  
> * **CriticalKV + GRKV:** average score = **41.51**【Source 2】.  
> 
> Thus, with GRKV the RULER average is 29.09 when using SnapKV and 41.51 when using CriticalKV.
>
> *Citations include "4 Experiments › 4.2 Experimental Results" (p. 7). Both numbers match Table 2.*

```
Ask Paper > Does GRKV require fine-tuning the model?
```
> *[Not grounded]* The retrieved passages do not answer this. The provided excerpts do not contain any statement indicating whether GRKV requires fine‑tuning of the language model. No source mentions model fin[…]
>
> *An honest miss: the paper says "training-free", but that passage wasn't retrieved for this wording. The model reported the gap instead of guessing, and the evaluation set tracks this type of miss.*

```
Ask Paper > What is the capital of France?
```
> *[Refused by the similarity gate: no chunk was similar enough, so the LLM was never called]*
> I cannot answer this question based on the paper. The document does not contain relevant information regarding[…]

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
| TF-IDF, gate 0.05 (run before table column headers were added) | 17/27 | 6/6 |
| Hybrid + reranker, before table column headers | 18/27 | 6/6 |
| **Hybrid + reranker, final code** | **21/27** | **6/6** |

**What this says, honestly:**
- Hybrid retrieval finds more evidence (20 vs 14 of 27). On the final code it answers 21 of 27 correctly, and it refuses every unanswerable and off-topic question.
- Table column headers mattered as well: the same hybrid setup went from 18 to 21, partly because the model stopped reading the wrong table column. The TF-IDF run came before that change, so part of the 17 → 21 gap comes from better table parsing, not only from retrieval.
- Of the 6 remaining misses, 5 are honest "the passages don't answer this" responses to paraphrased questions whose answer wasn't retrieved. One is an answer given from general knowledge (the attention-kernel question).
- 27 answerable questions is a small set, and the gate thresholds were chosen on it. Treat these numbers as evidence for the large effects (gate calibration, 14 → 20 retrieval, 17 → 21 answers), not for small differences.

---

## 6. Design Decisions & Tradeoffs

**Explicit graph without a framework.** Nodes, edges, routing conditions and entry points are declared as data in `build_research_graph()`. A small `StateGraph` class (about 120 lines) validates the graph, runs it, and renders the diagram. It uses the same model as LangGraph: typed shared state, conditional edges, several entry points. But at 9 nodes and one branch, it didn't justify a framework dependency, and every node stays a plain function that can be unit-tested. QA is a node on the same graph (entry point `ask`), so questions get the same logging, error routing and persistence as the analysis.

**State.** `AgentState` (Pydantic) holds the query, candidates, parsed sections, chunks with embeddings, briefing, QA history and a per-node log. It is saved as JSON after the graph and after every QA turn, so `--session` resumes a conversation without re-parsing or re-embedding.

**Retrieval: measured, then chosen.** I built a 33-question evaluation set before changing retrieval. Its first finding was a miscalibrated gate: TF-IDF at 0.15 refused 8 of 27 answerable questions. Off-topic questions score exactly 0 once stopwords are removed, so 0.05 fixed that. Embeddings alone ranked *worse* than TF-IDF on these number-heavy papers, but fusing both rankings (RRF) and adding a cross-encoder reranker found the answer for 20/27 questions, against 14/27. The models are an optional extra, so the base install stays light.

**Grounding in layers.** Paper metadata is copied from arXiv by code, never from the LLM. A similarity gate refuses off-topic questions before any LLM call. The QA prompt allows only retrieved passages and requires a `NOT IN PAPER:` marker for refusals, which the code checks. Tables are rebuilt as `row | header: value` so numbers stay attached to their row and column. The summarizer reads prose only, after one run mixed up numbers from flattened tables.

**Failure cases.** Zero arXiv results → the query is relaxed step by step. Many results → the LLM ranks them, preferring recent papers when asked. A PDF that won't download or has no text → the abstract, with a warning. A PyMuPDF failure → `pypdf`. LLM errors → retries and Gemini model fallback; malformed JSON gets one corrective re-ask; total failure yields an abstract-only briefing that says so.

**With more time:** grow the evaluation set (33 questions only shows large effects, and the thresholds were tuned on it); a second retrieval pass with a rewritten query when the LLM answers `NOT IN PAPER`; OCR for scanned PDFs.

**Known limitations.**
- Retrieval still misses about a quarter of answerable eval questions, mostly paraphrases answered by a single sentence. The agent then correctly says the passages don't answer, which isn't useful to the reader.
- When retrieval misses, the model occasionally answers from general knowledge anyway (one case in the evaluation).
- On-topic questions the paper doesn't answer pass the gate and rely on the LLM's refusal rule.
- Heading detection assumes LaTeX-style bold numbered headings. Floating tables can land in a neighbouring section, and sub-tables far below their header row lose their column names.
- The example briefing contains one wrong method detail. A briefing is a guide to reading the paper, not a substitute for it.

---

## 7. Running Tests

62 offline tests use a mock LLM, a fake embedder and a generated PDF, so no network, API keys or model downloads are needed. CI runs the same checks on every push (Python 3.10 and 3.13):

```bash
pytest tests/ -q                                   # 62 passed
ruff check arxiv_digest evals tests                # lint (pyflakes, bugbear, import order, ...)
ruff format --check arxiv_digest evals tests       # formatting
mypy arxiv_digest evals                            # type checking
```

The tests cover:
- **Graph:** routing for topic vs. ID, error routing, the `ask` entry point, validation, the cycle guard, and README/diagram sync.
- **arXiv and input handling:** IDs and URLs in every common form, API error entries, query relaxation, and rejecting non-PDF downloads.
- **Parsing:** headings, table rows with column headers, and page-accurate chunks.
- **Retrieval:** TF-IDF and hybrid (gate, fusion, reranking).
- **QA:** grounded answers and refusals.
- **LLM layer:** provider selection, retry and model fallback, briefing validation, the corrective re-ask, and the abstract-only fallback.
- **CLI:** clean exits for a corrupt session file and for Ctrl-C.

---

## 8. Project Directory Structure

```
.
├── README.md
├── pyproject.toml               # dependencies, ruff and mypy config
├── .github/workflows/ci.yml     # lint, type-check and tests on every push
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
│   ├── graph.py                 # StateGraph (nodes, edges, runner) + the agent's graph
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
└── tests/                       # 62 offline tests
```

---

## License
MIT License. Built by **Jay Gautam** for the **8byte** assessment.
