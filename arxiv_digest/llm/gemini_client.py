"""Google Gemini LLM provider (free tier).

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import json
from arxiv_digest.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """Client for Google Gemini models via google-generativeai."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. Install via: pip install google-generativeai"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = self._client.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt if system_prompt else None,
            generation_config=generation_config if generation_config else None,
        )

        response = model.generate_content(prompt)
        return response.text or ""
