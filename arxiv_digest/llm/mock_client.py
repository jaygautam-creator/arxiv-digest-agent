"""Deterministic offline Mock LLM provider for zero-dependency local runs and testing.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
import re
from arxiv_digest.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic mock provider providing high-fidelity structured outputs offline."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        prompt_lower = prompt.lower()

        # 1. Paper ranking prompt
        if "rank" in prompt_lower or "candidate" in prompt_lower or "select" in prompt_lower:
            # Look for index numbers or return candidate index 0
            if json_mode:
                return json.dumps({
                    "selected_index": 0,
                    "rationale": "Directly addresses the primary research objective and methodology specified in the user topic."
                })
            return "0"

        # 2. Executive Briefing Generation
        if "executive briefing" in prompt_lower or "problem statement" in prompt_lower or "why this paper matters" in prompt_lower:
            # Extract paper title if present
            title_match = re.search(r"Title:\s*([^\n]+)", prompt)
            paper_title = title_match.group(1).strip() if title_match else "Analyzed Research Paper"

            briefing_dict = {
                "title": paper_title,
                "authors": ["Lead Researcher", "Contributing Author"],
                "arxiv_id": "mock.arxiv.id",
                "publish_date": "2024-01-01",
                "link": "https://arxiv.org/abs/mock.arxiv.id",
                "summary_plain_english": (
                    f"This paper introduces a transformative approach to addressing computational efficiency "
                    f"and memory bottlenecks in deep learning architectures. By rethinking representation and "
                    f"caching mechanisms, the authors achieve significant inference speedups without accuracy degradation."
                ),
                "problem_statement": (
                    "Modern large-scale models suffer from severe quadratic memory growth and high latency "
                    "during sequence generation, severely limiting deployment on resource-constrained hardware."
                ),
                "method_approach": [
                    "Formulates a sparse, adaptive compression policy targeting token redundancy.",
                    "Integrates an attention-guided dynamic pruning gate during runtime generation.",
                    "Employs selective kernel approximation to maintain long-range context fidelity."
                ],
                "key_results_claims": [
                    "Reduces peak memory utilization by up to 4.2x during extended context generation.",
                    "Maintains 99.4% task performance across standard LLM evaluation benchmarks (MMLU, GSM8k).",
                    "Demonstrates 2.8x throughput acceleration on consumer-grade GPU clusters."
                ],
                "limitations": [
                    "Requires hardware support for structured sparse matrix multiplication to realize maximum gains.",
                    "Evaluated primarily on autoregressive text generation, with limited exploration on multimodal inputs.",
                    "Sensitivity to ultra-high compression ratios where rare token associations may be dropped."
                ],
                "suggested_followup_questions": [
                    "How does this compression method generalize to speculative decoding workflows?",
                    "What is the quantitative latency impact when switching between dense and compressed layers?",
                    "Can the dynamic pruning thresholds be learned end-to-end via reinforcement learning?"
                ]
            }
            return json.dumps(briefing_dict)

        # 3. QA Grounded Response
        if "question" in prompt_lower or "answer based only on" in prompt_lower:
            # Check if there is context provided
            context_match = re.search(r"Context:?\s*(.*?)(?:Question:|$)", prompt, re.DOTALL | re.IGNORECASE)
            context_text = context_match.group(1).strip() if context_match else ""

            # Check if out of scope / hallucination test
            if "capital of france" in prompt_lower or "weather" in prompt_lower or "unrelated" in prompt_lower:
                return (
                    "I cannot answer this question based on the provided paper. "
                    "The paper does not mention or cover this topic, and my answers are strictly grounded in the document text."
                )

            if context_text and len(context_text) > 20:
                first_sentence = context_text.split(".")[0].strip()
                return (
                    f"Based on the paper: {first_sentence}. "
                    f"The findings demonstrate consistent improvements aligned with the proposed architecture."
                )

            return "The paper does not contain sufficient information to answer this specific query."

        # Default fallback
        if json_mode:
            return json.dumps({"status": "ok", "message": "Processed successfully"})
        return "Mock response generated successfully."
