# 4-Minute Video Reflection Script

**Presenter:** Jay Gautam (<jaygautam561@gmail.com>)  
**Project:** Autonomous arXiv Paper Digest & QA Agent  
**Context:** 8byte Engineering Assessment  
**Target Duration:** Exactly 3 minutes 50 seconds (Safely under the 4-minute strict cutoff)  

---

## Recording Instructions for Jay
- **Camera/Screen:** Record your face in the corner with your screen showing the terminal running `arxiv-digest` and the architecture diagram in `README.md`.
- **Pacing:** Speak with steady, confident pacing. Follow the timing marks below.

---

### [0:00 – 0:35] 1. Introduction & System Vision
> *"Hi everyone, my name is Jay Gautam, and this is my submission for the 8byte engineering assessment: the Autonomous arXiv Paper Digest & QA Agent.*
>
> *When researchers and AI engineers evaluate literature, skimming through dozens of complex 15-page papers is a massive time sink. My goal with this project was to build a reliable, production-grade research assistant that transforms a raw research topic or an arXiv ID into an executive-level technical briefing, followed by a grounded, interactive Q&A session where you can probe the paper with zero risk of hallucinations.*
>
> *Rather than relying on a brittle monolithic prompt, I designed this as an explicit stateful graph with identifiable state that persists across stages and handles real-world failures gracefully."*

---

### [0:35 – 1:30] 2. The Stateful Graph Architecture & State Pipeline
> *"Let's look at the core pipeline. The system coordinates seven distinct nodes around a shared Pydantic `AgentState`:*
>
> 1. *First, **Query Understanding** parses whether the user provided a direct arXiv ID, a paper URL, or a natural-language research topic like 'recent work on KV-cache compression'.*
> 2. *Second, **arXiv Retrieval** queries the official arXiv Atom XML feed. To make this rock-solid, I implemented query tokenization that strips conversational stopwords like 'recent' and 'work' so the boolean search hits high-relevance papers.*
> 3. *Third, in topic searches, the **Paper Ranking** node evaluates candidate titles and abstracts using a composite score of lexical relevance and LLM semantic judgment to pick the single most impactful paper.*
> 4. *Fourth, **Fetch & Parse** streams the PDF with local disk caching and uses PyMuPDF to extract structural sections—preserving headings like Introduction, Methodology, Results, and Limitations.*
> 5. *Fifth, our **Section-Aware Chunker** splits text strictly within section boundaries with sliding sentence overlap, tagging each chunk with its exact section heading and page number.*
> 6. *Sixth, the chunks are indexed into a persistent **Local Vector Store** using sublinear TF-IDF and cosine similarity, completely eliminating external cloud database dependencies.*
> 7. *Finally, the **Summarizer** synthesizes an Executive Briefing strictly formatted into JSON and Markdown."*

---

### [1:30 – 2:25] 3. Correctness, Grounding & Anti-Hallucination Guard
> *"One of the highest-weight items in the rubric is correctness and preventing hallucinations in QA mode.*
>
> *Here is how I solved that: when a user asks a question, our retrieval engine queries the local index and calculates cosine similarity across chunks. I built a strict **Similarity Threshold Gate**: if an adversarial or out-of-domain question is asked—like 'What is the capital of France?'—the retrieval score falls below the cutoff threshold.*
>
> *Instead of letting the LLM guess, the agent triggers an immediate refusal gate: it explicitly informs the user that the paper does not contain information on this topic. When the question is within scope, the agent answers exclusively from retrieved snippets, citing the exact section and page number, such as `[Section 3.2, p. 5]`. This guarantees zero hallucinations."*

---

### [2:25 – 3:15] 4. Pragmatic Engineering & Failure Handling
> *"Under the hood, I prioritized pragmatic engineering and zero-cost accessibility:*
>
> - *First, **Zero Paid API Keys**: The agent works seamlessly with Google Gemini's free tier, Groq, local Ollama, and includes a deterministic offline Mock provider so the entire test suite and CLI run out of the box in automated environments.*
> - *Second, **arXiv CDN Resilience**: arXiv often rejects automated HTTP clients with 406 Not Acceptable errors. I implemented custom socket headers and automatic URL fallback logic to ensure downloads never fail silently.*
> - *Third, **Corrupted or Scanned PDFs**: If a PDF is a scanned bitmap with fewer than 300 characters, the agent catches it, logs a warning to the state, and supplements the index with the validated arXiv abstract so the pipeline never crashes.*
> - *Fourth, **Session Persistence**: The graph serializes its state to a local JSON session file, allowing users to reload previous analyses without re-downloading or re-indexing."*

---

### [3:15 – 3:50] 5. Tradeoffs & What I’d Build With More Time
> *"In terms of tradeoffs: I chose a lightweight, self-contained local vector store using NumPy over heavyweight libraries like Chroma or FAISS. This ensured instant setup on any machine without C++ compilation issues or 1GB PyTorch downloads.*
>
> *With more time, I would extend this in three ways:*
> 1. *Incorporate **hybrid ColBERT late-interaction embeddings** for finer token-level retrieval.*
> 2. *Add **multimodal figure and table parsing** using Vision-Language Models to summarize complex architecture diagrams.*
> 3. *Implement **iterative citation-graph traversal**, allowing the agent to follow related citations across multiple papers automatically.*
>
> *Thank you for reviewing my project, and I look forward to your feedback!"*

---
*(Total speaking duration: ~3 minutes 45 seconds)*
