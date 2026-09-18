"""Google Gemini LLM provider (free tier), via the REST API.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging

from arxiv_digest.llm.base import BaseLLM, LLMError, post_with_retry

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiLLM(BaseLLM):
    """Client for Google Gemini models.

    Newer Flash models are frequently overloaded on the free tier (HTTP 503). A model
    that has a fallback gets a single short attempt; once it fails it is skipped for the
    rest of the session, so one overloaded model does not stall every later call.
    The last model in the chain gets the full retry budget.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
        fallback_models: list[str] | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model_name]
        self.last_model_used: str | None = None
        self._unavailable: set[str] = set()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if json_mode:
            payload["generationConfig"] = {"responseMimeType": "application/json"}

        chain = [m for m in [self.model_name, *self.fallback_models] if m not in self._unavailable]
        chain = chain or [self.fallback_models[-1] if self.fallback_models else self.model_name]
        errors = []
        for position, model in enumerate(chain):
            has_fallback = position < len(chain) - 1
            try:
                data = post_with_retry(
                    f"{API_BASE}/{model}:generateContent",
                    payload,
                    params={"key": self.api_key},
                    timeout=60.0 if has_fallback else 90.0,
                    max_attempts=1 if has_fallback else 3,
                )
            except LLMError as e:
                logger.warning("Gemini model %s unavailable (%s); trying next model.", model, str(e)[:120])
                errors.append(f"{model}: {e}")
                if has_fallback:
                    self._unavailable.add(model)
                continue
            self.last_model_used = model
            return _extract_text(data)

        raise LLMError("All Gemini models failed. " + " | ".join(errors))


def _extract_text(data: dict) -> str:
    """Concatenate text parts of the first candidate, skipping thought parts."""
    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates returned")
        raise LLMError(f"Gemini returned no output ({reason})")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))
