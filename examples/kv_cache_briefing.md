# Executive Briefing: GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs

**Authors:** Junjie Peng, You Wu, Haoyi Wu, Jialong Han, Xiaohua Xie, Kewei Tu, Jianhuang Lai  
**arXiv ID:** [2605.31105](https://arxiv.org/abs/2605.31105v2) | **Published:** 2026-05-29

---

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
