"""Base interface for LLM providers.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
import time
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)

# Rate limits (429) and transient overload (5xx) are expected on free tiers.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


class BaseLLM(ABC):
    """Abstract LLM wrapper ensuring interchangeable providers."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generate text from prompt."""


def post_with_retry(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: float = 90.0,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> dict:
    """POST JSON, retrying rate-limit and overload errors with exponential backoff.

    Honors a numeric `Retry-After` header when the server sends one.
    """
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        resp: httpx.Response | None = None
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=headers, params=params)
        except httpx.TransportError as e:
            last_error = f"network error: {e}"
        else:
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {_error_message(resp)}"
            if resp.status_code not in RETRYABLE_STATUS:
                raise LLMError(last_error)

        if attempt < max_attempts:
            delay = base_delay * 2 ** (attempt - 1)
            retry_after = resp.headers.get("retry-after", "") if resp is not None else ""
            if retry_after.replace(".", "", 1).isdigit():
                delay = max(delay, float(retry_after))
            logger.warning("LLM call failed (%s); retry %d/%d in %.0fs", last_error, attempt, max_attempts - 1, delay)
            time.sleep(delay)

    raise LLMError(f"Gave up after {max_attempts} attempts: {last_error}")


def _error_message(resp: httpx.Response) -> str:
    """First line of the provider's error message, without the raw JSON envelope."""
    try:
        message = resp.json().get("error", {}).get("message", "")
    except ValueError:
        message = resp.text
    return (message or resp.reason_phrase).strip().splitlines()[0][:160]
