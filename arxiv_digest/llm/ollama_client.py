"""Local Ollama LLM provider (free and offline).

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import httpx
from arxiv_digest.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Client for locally hosted Ollama instances."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
