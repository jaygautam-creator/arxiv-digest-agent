# Example Run: Autonomous arXiv Paper Digest & QA Agent

**Target Query:** `"recent work on KV-cache compression for LLMs"`  
**Selected Paper:** *PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference*  
**Date:** 2026-09-18  
**Author:** Jay Gautam (<jaygautam561@gmail.com>) for 8byte  

---

## 1. Pipeline Execution Trace

```bash
$ arxiv-digest "recent work on KV-cache compression for LLMs"

╭──────────────────────────────────────────────────────────────────────────────╮
│  Autonomous arXiv Paper Digest & QA Agent                                    │
│  Stateful Research Graph  |  Authored by Jay Gautam for 8byte                │
╰──────────────────────────────────────────────────────────────────────────────╯
Active Provider: gemini
Target Query: recent work on KV-cache compression for LLMs

[query_understanding] Detected topic search intent: 'KV-cache compression LLMs'
[arxiv_retrieval] Found 5 candidate paper(s) on arXiv
[paper_ranking] Ranked 5 candidates. Selected: 'PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference'
[fetch_and_parse] Downloaded PDF and extracted 8 structural sections
[chunk_and_embed] Generated 34 section-aware semantic chunks with page provenance
[vector_indexing] Indexed into LocalVectorStore (Vocabulary: 1,420 terms)
[summarize_briefing] Executive briefing synthesized successfully
```

---

## 2. Generated Executive Briefing

```markdown
# Executive Briefing: PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference

**Authors:** Chen Zhang, Mingyu Gao, Lingxiao Ma, Fan Yang  
**arXiv ID:** [2604.24971](https://arxiv.org/abs/2604.24971) | **Published:** 2026-04-28

---

## 1. Why This Paper Matters
In multi-agent LLM systems, concurrent agents redundantly process shared context (e.g. system prompts, world states, tool descriptions), leading to prohibitive KV-cache memory explosions. PolyKV addresses this with an asymmetric, shared KV-cache pool across agents, slashing memory pressure without compromising task completion accuracy.

## 2. Problem Statement
Existing KV cache compression methods operate independently per-session. In collaborative multi-agent settings, this causes duplicate retention of identical prefix tokens across distinct agent contexts, causing GPU memory exhaustion and bottlenecking batch concurrency.

## 3. Method & Technical Approach
- Implements a unified prefix-sharing cache pool with hierarchical token reference counting.
- Employs asymmetric compression: retains high-precision keys/values for agent-shared instructions while aggressively quantizing agent-private reasoning tokens.
- Introduces dynamic cache eviction gates based on inter-agent token reuse frequency.

## 4. Key Results & Claims
- Achieves up to 3.8x KV-cache memory reduction across 8-agent collaborative benchmarks.
- Maintains 99.2% benchmark accuracy compared to uncompressed multi-agent baselines.
- Increases multi-agent inference throughput by 2.4x on 8x A100 GPU clusters.

## 5. Limitations & Edge Cases
- Memory reduction benefits diminish when agent workflows exhibit zero prefix or context sharing.
- Asymmetric quantization requires specialized runtime kernels for mixed-precision matrix multiplication.
- Evaluated primarily on cooperative role-playing and code-generation agent benchmarks.

## 6. Suggested Follow-up Questions
- How does PolyKV coordinate cache eviction when agents possess divergent context retention priorities?
- What is the latency penalty of cross-agent lock contention in the shared cache pool?
- Can the asymmetric quantization thresholds be adapted dynamically based on GPU thermal throttling?
```

---

## 3. Sample Grounded QA Exchanges

### Exchange 1: Technical Architecture
**User Query:**
> *"How does the paper achieve asymmetric compression between shared and private tokens?"*

**Agent Response:**
> Based on Section 3 (*System Architecture*, pp. 4–5), PolyKV segregates the KV cache into a global shared prefix segment and agent-specific private segments. Shared prefix tokens are stored with high bit-width precision (FP16/BF16) to prevent compounding degradation across agents. In contrast, private reasoning tokens undergo aggressive 4-bit grouped quantization with dynamic outlier protection `[Section 3.2, p. 5]`.

**Retrieved Citations:**
| Section | Page | Relevance | Excerpt Snippet |
|---|---|---|---|
| Section 3.2: Asymmetric Cache | 5 | 0.84 | "Shared prefix tokens maintain 16-bit representation, whereas agent-private branches are quantized to INT4..." |
| Section 3.1: Pool Hierarchy | 4 | 0.72 | "Hierarchical reference counting tracks inter-agent access frequencies to prevent premature eviction..." |

---

### Exchange 2: Benchmark Results & Speedup
**User Query:**
> *"What throughput acceleration was measured on multi-agent benchmarks?"*

**Agent Response:**
> The experimental results in Section 5 (*Evaluation*, p. 8) report a 2.4x throughput increase on an 8-agent benchmark running on an 8x A100 GPU cluster. This acceleration stems directly from the reduced GPU memory footprint, enabling larger active batch sizes without out-of-memory (OOM) faults `[Section 5.3, p. 8]`.

**Retrieved Citations:**
| Section | Page | Relevance | Excerpt Snippet |
|---|---|---|---|
| Section 5.3: Throughput & Scaling | 8 | 0.89 | "Throughput scaled by 2.4x under concurrent 8-agent workloads by increasing max batch size from 4 to 12..." |

---

### Exchange 3: Anti-Hallucination Guard Test (Out-of-Domain Query)
**User Query:**
> *"What is the capital of France and how does the weather affect the model?"*

**Agent Response:**
> *[Ungrounded Query Refusal]*  
> **"I cannot answer this question based on the paper. The document does not contain relevant information regarding this query, and answers are strictly restricted to verified source content to prevent hallucination."**

**Grounding Status:** `False` (0 chunks above similarity threshold; hallucination strictly prevented).
