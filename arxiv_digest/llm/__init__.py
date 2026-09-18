"""LLM provider factory and registry.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import logging
from arxiv_digest.config import AgentConfig, LLMProviderType
from arxiv_digest.llm.base import BaseLLM
from arxiv_digest.llm.mock_client import MockLLM

logger = logging.getLogger(__name__)


def get_llm_provider(config: AgentConfig) -> BaseLLM:
    """Instantiate the configured LLM provider with graceful fallback."""
    if config.provider == LLMProviderType.GEMINI:
        if not config.gemini_api_key:
            logger.warning("GEMINI_API_KEY not provided. Falling back to MockLLM.")
            return MockLLM()
        try:
            from arxiv_digest.llm.gemini_client import GeminiLLM
            return GeminiLLM(api_key=config.gemini_api_key, model_name=config.gemini_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client ({e}). Falling back to MockLLM.")
            return MockLLM()

    elif config.provider == LLMProviderType.GROQ:
        if not config.groq_api_key:
            logger.warning("GROQ_API_KEY not provided. Falling back to MockLLM.")
            return MockLLM()
        try:
            from arxiv_digest.llm.groq_client import GroqLLM
            return GroqLLM(api_key=config.groq_api_key, model_name=config.groq_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client ({e}). Falling back to MockLLM.")
            return MockLLM()

    elif config.provider == LLMProviderType.OLLAMA:
        try:
            from arxiv_digest.llm.ollama_client import OllamaLLM
            return OllamaLLM(base_url=config.ollama_base_url, model_name=config.ollama_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client ({e}). Falling back to MockLLM.")
            return MockLLM()

    return MockLLM()
