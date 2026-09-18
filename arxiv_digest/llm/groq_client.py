"""Groq LLM provider (free tier).

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.llm.base import BaseLLM


class GroqLLM(BaseLLM):
    """Client for Groq fast inference models."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model_name = model_name
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("groq package is not installed. Install via: pip install groq")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_format = {"type": "json_object"} if json_mode else None

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
