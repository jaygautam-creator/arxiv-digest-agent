# Example Run: Topic Query → Briefing → Grounded QA

**Author:** Jay Gautam (<jaygautam561@gmail.com>) for 8byte
**Recorded:** 2026-09-18 · **Provider:** Groq `openai/gpt-oss-120b` (free tier)

Everything below is unedited output from a real run. It was recorded with TF-IDF retrieval (gate 0.15), before hybrid retrieval, table-row reconstruction and exact page ranges were added; see the README's Retrieval Evaluation for how those change QA. The briefing is also
saved as [`kv_cache_briefing.md`](kv_cache_briefing.md) / [`kv_cache_briefing.json`](kv_cache_briefing.json).
arXiv search results change over time, so re-running the same topic may select a different paper.

---

## 1. Input and pipeline trace

```bash
$ LLM_PROVIDER=groq python -m arxiv_digest "recent work on KV-cache compression for LLMs"
```

Per-node log from the saved session (`AgentState.execution_logs`):

```
[query_parser]       success     0 ms  Detected topic search intent
[arxiv_retrieval]    success   237 ms  Found 5 candidate paper(s)
[paper_ranking]      success  1735 ms  Selected paper [5/5]: 'GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs'
[fetch_and_parse]    success  2107 ms  Parsed paper into 33 sections and 97329 characters
[chunk_and_embed]    success     3 ms  Generated 137 section-aware chunks across 33 sections
[vector_indexing]    success    12 ms  Indexed 137 chunks into LocalVectorStore (Vocabulary: 3016 terms)
[summarize_briefing] success  3908 ms  Generated executive briefing
```

### Candidates returned by arXiv

| arXiv ID | Published | Title |
|---|---|---|
| 2604.24971 | 2026-04-27 | PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference |
| 2602.05929 | 2026-02-05 | KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs |
| 2512.14946 | 2025-12-16 | EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving |
| 2410.03111 | 2024-10-04 | LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy |
| **2605.31105** | **2026-05-29** | **GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs** |

**Selection rationale (from the ranking node):** "GRKV presents a training‑free, regression‑based KV‑cache
compression technique for long‑context LLMs, offering the most recent and novel direct solution to KV‑cache
memory reduction."

---

## 2. Executive briefing

```markdown
# Executive Briefing: GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs

**Authors:** Junjie Peng, You Wu, Haoyi Wu, Jialong Han, Xiaohua Xie, Kewei Tu, Jianhuang Lai
**arXiv ID:** [2605.31105](https://arxiv.org/abs/2605.31105v2) | **Published:** 2026-05-29

## 1. Why This Paper Matters
This paper tackles the memory bottleneck of long‑context LLM inference by introducing a training‑free KV‑cache merging technique that directly optimizes the attention output discrepancy between a compressed cache and the full cache. By formulating the merge as a ridge‑regression problem over all retained tokens, GRKV recovers information lost during span‑based eviction without costly retraining, delivering consistent accuracy gains across multiple open‑source models while adding only minimal compute overhead—making it immediately useful for engineers deploying large models with limited GPU memory.

## 2. Problem Statement
Span‑based KV‑cache eviction concentrates merges onto a few boundary tokens, creating an imbalanced merge pattern that over‑merges and discards contextual information, leading to degraded attention quality under strict cache‑budget constraints (e.g., 10% of the context).

## 3. Method & Technical Approach
- Uses a surrogate query window derived from the prompt to approximate future queries and defines a global regression objective that minimizes the cosine‑distance between compressed‑cache and full‑cache attention outputs.
- Solves a ridge‑regularized linear least‑squares problem for values (closed‑form solution) and a local linearized ridge update for keys, alternating between them (one update step by default).
- Employs the Woodbury identity to solve the regression in the low‑dimensional query‑window space (m≈32) instead of the full cache dimension (c≈10% of context length), drastically reducing compute and memory cost.

## 4. Key Results & Claims
- On Llama-3.1-8B-Instruct (LongBench, 10% cache budget) GRKV raises the average score from 33.96 to 34.58 with SnapKV and from 36.00 to 36.58 with CriticalKV, improving 14/16 tasks in both cases.
- On Mistral-7B-Instruct‑v0.3 (LongBench, 10% cache budget) GRKV improves SnapKV from 33.12 to 33.75 and CriticalKV from 33.69 to 34.30, with gains on 12/16 and 14/16 tasks respectively. Regularization λk=λv=10⁻² yields the best LongBench average (34.58) on Llama-3.1-8B‑Instruct with SnapKV (10% budget); removing regularization drops the score to 32.06, while λ=1 and λ=10⁻¹ give 34.33 and 34.29 respectively. Fixing a retained‑token ratio β=10% also attains 34.58, whereas fixing none yields 34.34.

## 5. Limitations & Edge Cases
- Evaluation is limited to three open‑source English models (Llama‑3.1‑8B‑Instruct, Mistral‑7B‑Instruct‑v0.3, Qwen3‑14B) and two long‑context benchmarks (LongBench, RULER); results may not transfer to larger proprietary models or multilingual/multimodal settings.
- The surrogate prompt‑derived query window is an empirical proxy; misalignment between this window and actual future queries can reduce reconstruction effectiveness, especially when pre‑fill latency dominates or when the cache budget is increased.

## 6. Suggested Follow-up Questions
- How does GRKV perform on proprietary, higher‑parameter models (e.g., 70B‑plus) and on multilingual long‑context tasks?
- Can the surrogate window be learned or adapted per head to improve alignment with diverse future query distributions?
- What is the trade‑off between the number of alternating KV update steps (S) and inference latency in real‑world deployment, and can adaptive stopping criteria mitigate over‑fitting?
```

### Manual verification against the PDF

I checked every factual claim against the paper text:

- **Metadata** (title, authors, date, link) is copied from the arXiv API by code, never taken from the LLM.
- **All 20 numbers** in "Key Results" match the paper, with the correct model, benchmark and base method:
  §4.2 prose for the LongBench scores, Table 3 for the λ and β ablations.
- **One error:** the first Method bullet says the objective minimizes "cosine‑distance". The paper minimizes
  the **squared L2** discrepancy `‖o_full − o_ret‖²` (Eq. 1–2). The paper uses cosine similarity only in its
  analysis (window-consistency heatmaps) and in the baselines' key-matching rule, never in GRKV's objective.
  I left the output unedited so it reflects what the tool actually produces.

---

## 3. Grounded QA exchanges

Answers come from `python -m arxiv_digest --session data/sessions/session_7dd92a50.json`. Each lists the chunks
retrieved for it (section path, page, cosine score).

### Q1. How does GRKV use ridge regression to merge evicted tokens?

> GRKV merges evicted tokens by solving a ridge‑regression problem that aligns the compressed KV cache with the original (full) cache. In each merge step the full cache is taken as the information source, and a ridge‑regression objective is minimized so that the updates to the retained (carrier) tokens distribute the evicted‑token information across all retained tokens while a regularization term limits the magnitude of the updates. This regularization prevents over‑smoothing and "carrier‑token blurring," thereby reducing information loss during the merge [Source 1]; the same ridge‑regression regularization is highlighted as the mechanism that mitigates over‑merging in GRKV [Source 4].

| Section | Page | Score |
|---|---|---|
| 1 Introduction | 2 | 0.265 |
| 1 Introduction | 1 | 0.215 |
| A Why Span-Based Retention Amplifies › A.2 Local KV Merging and Carrier Support | 12 | 0.171 |
| 1 Introduction | 3 | 0.170 |

### Q2. What does GRKV score on RULER with Llama-3.1-8B-Instruct at a 10% cache budget?

> GRKV attains an average RULER score of **29.09** with Llama‑3.1‑8B‑Instruct when the cache budget is limited to 10 % (using the SnapKV eviction method)【Source 2】. (With the alternative CriticalKV eviction method, the score is 41.51【Source 2】.)

| Section | Page | Score |
|---|---|---|
| F Additional Results on LongBench and RULER at 20% Cache Budget | 21 | 0.354 |
| 4 Experiments › 4.2 Experimental Results | 7 | 0.318 |
| G Additional Results on Ablations, Compatibility, and Efficiency | 22 | 0.306 |
| F Additional Results on LongBench and RULER at 20% Cache Budget | 21 | 0.287 |

Both numbers match Table 2 of the paper. The top-ranked chunk is labelled with appendix section F even
though it is Table 13 from appendix G: floating tables can land just before the next heading in the PDF's
reading order.

### Q3. Does GRKV require fine-tuning the model? (retrieval miss)

> The retrieved passages do not answer this. The provided excerpts do not contain any statement about whether GRKV requires fine‑tuning of the language model. None of the cited sections mention model fine‑tuning or explicitly state that GRKV operates without it.

Flagged as not grounded (`is_grounded=False`). The paper does answer it, with "training-free" in its title and
abstract, but lexical TF-IDF retrieval cannot match "fine-tuning" to "training-free". The model reported the gap
rather than guessing, which is the intended failure mode. Dense embeddings would fix this; see the README.

### Q4. What is the capital of France? (out of scope)

> I cannot answer this question based on the paper. The document does not contain relevant information regarding this query, and answers are strictly restricted to verified source content to prevent hallucination.

The best chunk scored 0.0, below the 0.15 similarity threshold, so the gate refused before any LLM call.
