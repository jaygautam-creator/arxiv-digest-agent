# Example Run: Topic Query → Briefing → Grounded QA

**Author:** Jay Gautam (<jaygautam561@gmail.com>) for 8byte
**Recorded:** 2026-09-19 · **LLM:** Groq `openai/gpt-oss-120b` (free tier) · **Retrieval:** hybrid (bge-small + TF-IDF, MiniLM reranker)

Unedited output from a real run of the current code. The briefing is also saved as
[`kv_cache_briefing.md`](kv_cache_briefing.md) / [`kv_cache_briefing.json`](kv_cache_briefing.json).
arXiv search results change over time, so re-running the same topic may select a different paper.

---

## 1. Input and graph trace

```bash
$ python -m arxiv_digest "recent work on KV-cache compression for LLMs"
```

Path through the graph (`AgentState.visited_nodes`, as saved in the session file):
`query_understanding → arxiv_retrieval → paper_ranking → fetch_and_parse → chunk_and_embed → vector_indexing → summarize_briefing → persist_session`

```
[query_understanding] success      0 ms  Detected topic search intent: 'recent work on KV-cache compression for LLMs'
[arxiv_retrieval    ] success    152 ms  Found 5 candidate paper(s) for topic: 'recent work on KV-cache compression for LLMs'
[paper_ranking      ] success   1383 ms  Selected paper [5/5]: 'GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs' - 
[fetch_and_parse    ] success   2150 ms  Parsed paper into 33 sections and 113022 characters.
[chunk_and_embed    ] success      3 ms  Generated 159 section-aware chunks across 33 sections.
[vector_indexing    ] success  21006 ms  Indexed 159 chunks (hybrid+rerank retrieval, vocabulary 3017 terms).
[summarize_briefing ] success   3290 ms  Generated executive briefing for 'GRKV: Global Regression for Training-Free KV Cache Compression in Long-Conte
```

`vector_indexing` takes most of the time because it embeds 159 chunks on the CPU; later runs on the same session reuse the stored embeddings.

### Candidates returned by arXiv

| arXiv ID | Published | Title |
|---|---|---|
| 2604.24971 | 2026-04-27 | PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference |
| 2602.05929 | 2026-02-05 | KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs |
| 2512.14946 | 2025-12-16 | EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving |
| 2410.03111 | 2024-10-04 | LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy |
| **2605.31105** | 2026-05-29 | GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs |

**Ranking rationale:** "GRKV (2026-05-29) introduces a training‑free, globally‑regressed compression method targeting long‑context LLMs, directly advancing KV‑cache compression techniques and being the most recent work among the candidates."

---

## 2. Executive briefing

```markdown
# Executive Briefing: GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs

**Authors:** Junjie Peng, You Wu, Haoyi Wu, Jialong Han, Xiaohua Xie, Kewei Tu, Jianhuang Lai  
**arXiv ID:** [2605.31105](https://arxiv.org/abs/2605.31105v2) | **Published:** 2026-05-29

---

## 1. Why This Paper Matters
The paper introduces GRKV, a training‑free method that merges KV‑cache entries by solving a ridge‑regression problem, directly minimizing the difference between attention outputs of a compressed cache and the full cache. By distributing information from evicted tokens across retained tokens, GRKV eliminates the over‑merging bias of span‑based eviction and consistently improves downstream long‑context performance with negligible compute overhead, making it immediately useful for engineers deploying large language models with limited memory budgets.

## 2. Problem Statement
Span‑based KV‑cache eviction concentrates merges onto a few boundary tokens, creating an imbalanced merge pattern that over‑merges and loses information, especially under tight cache budgets (e.g., 10% of the context). Existing merging heuristics cannot fully recover the discarded context and may even degrade performance.

## 3. Method & Technical Approach
- Uses a surrogate query window (default length m=32) derived from the prompt to approximate future queries during generation
- Solves a ridge‑regularized linear regression (ridge‑regression) for values and a linearized ridge update for keys, alternating one update step (S=1) to reconstruct full‑cache attention outputs
- Applies the Woodbury identity to solve the regression in the smaller m×m space, reducing compute and memory when the cache budget c is large

## 4. Key Results & Claims
- On Llama-3.1-8B‑Instruct with a 10% cache budget, GRKV raises the LongBench average score from 33.96 to 34.58 when paired with SnapKV and from 36.00 to 36.58 when paired with CriticalKV (improving 14/16 tasks in both cases)
- On Mistral-7B‑Instructv0.3 with a 10% cache budget, GRKV improves SnapKV’s LongBench average from 33.12 to 33.75 (12/16 tasks) and CriticalKV’s from 33.69 to 34.30 (14/16 tasks)
- Ablation shows that a mild ridge penalty (λk=λv=10⁻²) yields the best LongBench average of 34.58, whereas removing regularization drops the score to 32.06
- Using the default retained‑token ratio β=10% gives 34.58, while fixing no retained tokens gives 34.34
- Surrogate window size m=32 achieves 34.58; increasing to m=48 reduces the average to 33.86 and to m=64 reduces it further to 33.49

## 5. Limitations & Edge Cases
- Evaluations are limited to three open‑source models (Llama‑3.1‑8B‑Instruct, Mistral‑7B‑Instructv0.3, Qwen3‑14B) and two English‑centric long‑context benchmarks (LongBench and RULER); results may not generalize to larger proprietary models, multilingual or multimodal settings
- The surrogate query window is an empirical proxy; misalignment between surrogate and actual future queries can diminish reconstruction benefits, and the method does not guarantee uniform gains across all heads, samples, or tasks

## 6. Suggested Follow-up Questions
- How does GRKV perform when the cache budget is increased (e.g., 20% or 30%) or when prefill latency dominates inference cost?
- Can the ridge‑regularized reconstruction be extended to multi‑head attention jointly, and does that yield further gains?
- What is the impact of GRKV on downstream tasks that require fine‑grained token‑level fidelity (e.g., code generation) compared to tasks focused on high‑level semantics?
```

### Checked against the PDF

- **Metadata** (title, authors, date, link) is copied from the arXiv API by code, never taken from the LLM.
- **All 24 numbers** in the briefing appear in the paper, attributed to the right model, benchmark and base method: the headline results are LongBench (§4.2 prose), and the ablations are Table 3.
- **One overstatement:** "Why This Paper Matters" says GRKV *eliminates* the over-merging bias. The paper says it *mitigates* over-merging. I left the output unedited.

An earlier run of this pipeline attributed the LongBench numbers to RULER, because a floating RULER table and its caption sat inside the LongBench paragraph in the PDF's text order. The summarizer now removes tables, captions and labels from its context and re-joins words they split, which fixed it.

---

## 3. Grounded QA exchanges

The questions were asked with `python -m arxiv_digest --session data/sessions/session_8d49b4a7.json`. Each answer ran the graph's `ask` entry point (`answer_question → persist_session`). Citations list the retrieved chunks with section path, page range and embedding similarity.

### Q1. How does GRKV use ridge regression to merge evicted tokens?

> GRKV treats the merging of evicted tokens as a **global regression problem**: it seeks a set of updates to the retained (carrier) tokens that makes the attention output of the compressed cache match the attention output of the full cache as closely as possible.  To obtain these updates, GRKV solves a **ridge‑regression** problem in which the full cache provides the target information and the retained tokens serve as the regression variables (the “carriers”).  The ridge‑regression formulation distributes the information from each evicted token across all retained tokens while a regularization term penalizes large changes, thereby preventing over‑smoothing or carrier‑token blurring【Source 3】.  In practice, each merge step solves this ridge‑regression to compute the adjustments to the retained tokens, effectively merging the evicted tokens into the carriers in a way that directly minimizes the discrepancy between compressed‑cache and full‑cache attention outputs【Source 1】.

| Section | Page | Similarity |
|---|---|---|
| Abstract | 1 | 0.807 |
| 1 Introduction | 3 | 0.835 |
| 1 Introduction | 2–3 | 0.794 |
| 5 Conclusion | 8–9 | 0.768 |

Grounded, citing the abstract, introduction and conclusion; page ranges show where a passage spans two pages.

### Q2. What does GRKV score on RULER with Llama-3.1-8B-Instruct at a 10% cache budget?

> GRKV improves the RULER scores for Llama‑3.1‑8B‑Instruct under a 10 % cache budget:
> 
> * **SnapKV + GRKV:** average score = **29.09**【Source 2】 (also noted in the ablation table as the best average when λₖ = λᵥ = 10⁻²【Source 1】).  
> * **CriticalKV + GRKV:** average score = **41.51**【Source 2】.  
> 
> Thus, with GRKV the RULER average is 29.09 when using SnapKV and 41.51 when using CriticalKV.

| Section | Page | Similarity |
|---|---|---|
| G Additional Results on Ablations, Compatibility, and Efficiency | 22 | 0.801 |
| 4 Experiments › 4.2 Experimental Results | 7 | 0.815 |
| F Additional Results on LongBench and RULER at 20% Cache Budget | 21–22 | 0.834 |
| 4 Experiments › 4.2 Experimental Results | 7 | 0.839 |

Both numbers match Table 2 (Llama-3.1-8B-Instruct, 16K RULER, 10% budget: SnapKV + GRKV 29.09, CriticalKV + GRKV 41.51).

### Q3. Does GRKV require fine-tuning the model?

> The retrieved passages do not answer this. The provided excerpts do not contain any statement indicating whether GRKV requires fine‑tuning of the language model. No source mentions model fine‑tuning as a prerequisite for using GRKV.

Flagged as not grounded. The paper does answer this ("training-free" is in its title), but the passage wasn't retrieved for this wording. The model reported the gap instead of guessing. This is the most common failure type in the evaluation set.

### Q4. What is the capital of France?

> I cannot answer this question based on the paper. The document does not contain relevant information regarding this query, and answers are strictly restricted to verified source content to prevent hallucination.

Refused by the similarity gate: no chunk reached the threshold, so the LLM was never called.
