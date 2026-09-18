"""LLM provider factory and registry.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.llm.base import BaseLLM, LLMError
from arxiv_digest.llm.gemini_client import GeminiLLM
from arxiv_digest.llm.groq_client import GroqLLM
from arxiv_digest.llm.mock_client import MockLLM
from arxiv_digest.llm.ollama_client import OllamaLLM


class ProviderConfigError(ValueError):
    """Raised when the selected provider is missing required configuration."""


def get_llm_provider(config: AgentConfig) -> BaseLLM:
    """Instantiate the configured LLM provider.

    A misconfigured provider is an error rather than a silent switch to mock,
    so placeholder output can never be mistaken for a real model's.
    """
    if config.provider == LLMProviderType.GEMINI:
        if not config.gemini_api_key:
            raise ProviderConfigError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set (see .env.example).")
        return GeminiLLM(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            fallback_models=config.gemini_fallback_models,
        )

    if config.provider == LLMProviderType.GROQ:
        if not config.groq_api_key:
            raise ProviderConfigError("LLM_PROVIDER=groq but GROQ_API_KEY is not set (see .env.example).")
        return GroqLLM(api_key=config.groq_api_key, model_name=config.groq_model)

    if config.provider == LLMProviderType.OLLAMA:
        return OllamaLLM(base_url=config.ollama_base_url, model_name=config.ollama_model)

    return MockLLM()


def describe_provider(config: AgentConfig) -> str:
    """Human-readable provider/model label for CLI output."""
    models = {
        LLMProviderType.GEMINI: config.gemini_model,
        LLMProviderType.GROQ: config.groq_model,
        LLMProviderType.OLLAMA: config.ollama_model,
        LLMProviderType.MOCK: "offline placeholder output",
    }
    return f"{config.provider.value} ({models[config.provider]})"


__all__ = ["BaseLLM", "LLMError", "ProviderConfigError", "describe_provider", "get_llm_provider"]
