"""Base interface for LLM providers.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract LLM wrapper ensuring interchangeable providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generate text from prompt."""
        pass
