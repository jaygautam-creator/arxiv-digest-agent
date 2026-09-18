"""Groq LLM provider (free tier), via its OpenAI-compatible REST API.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.llm.base import BaseLLM, LLMError, post_with_retry

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM(BaseLLM):
    """Client for Groq fast inference models."""

    name = "groq"

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.api_key = api_key
        self.model_name = model_name
        self.last_model_used: str | None = None

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

        payload: dict = {"model": self.model_name, "messages": messages, "temperature": 0.2}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = post_with_retry(
            API_URL,
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Groq returned no choices")
        self.last_model_used = self.model_name
        return choices[0].get("message", {}).get("content") or ""
