from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from financialagents.config import Settings


def create_llm_client(
    tier: str,
    config: Settings,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    if tier not in ("deep_think", "quick_think"):
        raise ValueError(
            f"Invalid tier '{tier}'. Must be 'deep_think' or 'quick_think'."
        )

    model_name = config.deep_think_llm if tier == "deep_think" else config.quick_think_llm
    provider = config.llm_provider

    if provider == "groq":
        from financialagents.llm_clients.groq_client import create_groq_client

        return create_groq_client(
            model=model_name,
            api_key=config.groq_api_key,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )

    elif provider == "ollama":
        from financialagents.llm_clients.ollama_client import create_ollama_client

        return create_ollama_client(
            model=model_name,
            num_predict=max_output_tokens,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Valid options: 'groq', 'ollama'"
        )
