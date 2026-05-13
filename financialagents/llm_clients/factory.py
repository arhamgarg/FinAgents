from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from financialagents.config import Settings

if TYPE_CHECKING:
    from financialagents.testing.llm_mock._call_sites import CallSite
    from financialagents.testing.llm_mock._harness import LLMMockHarness

_ACTIVE_HARNESS: "LLMMockHarness | None" = None


def _set_active_harness(harness: "LLMMockHarness | None") -> None:
    global _ACTIVE_HARNESS
    _ACTIVE_HARNESS = harness


def create_llm_client(
    tier: str,
    config: Settings,
    temperature: float | None = None,
    call_site: "CallSite | None" = None,
) -> BaseChatModel:
    if _ACTIVE_HARNESS is not None:
        return _ACTIVE_HARNESS._fake_chat_model_for(call_site)

    if tier not in ("deep_think", "quick_think"):
        raise ValueError(
            f"Invalid tier '{tier}'. Must be 'deep_think' or 'quick_think'."
        )

    model_name = config.deep_think_llm if tier == "deep_think" else config.quick_think_llm
    provider = config.llm_provider

    if provider == "openrouter":
        from financialagents.llm_clients.openrouter_client import create_openrouter_client

        return create_openrouter_client(
            model=model_name,
            api_key=config.openrouter_api_key,
            temperature=temperature,
        )

    elif provider == "cerebras":
        from financialagents.llm_clients.cerebras_client import create_cerebras_client

        return create_cerebras_client(
            model=model_name,
            api_key=config.cerebras_api_key,
            temperature=temperature,
        )

    elif provider == "groq":
        from financialagents.llm_clients.groq_client import create_groq_client

        return create_groq_client(
            model=model_name,
            api_key=config.groq_api_key,
            temperature=temperature,
        )

    elif provider == "ollama":
        from financialagents.llm_clients.ollama_client import create_ollama_client

        return create_ollama_client(
            model=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            "Valid options: 'openrouter', 'cerebras', 'groq', 'ollama'"
        )
