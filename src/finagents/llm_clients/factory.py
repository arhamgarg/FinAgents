from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langchain_core.language_models import BaseChatModel

from finagents.config import Settings, resolve_model

CallSite = Literal[
    "technical_analyst",
    "fundamentals_analyst",
    "news_analyst",
    "research_manager",
    "bull_researcher",
    "bear_researcher",
    "investment_thesis",
    "trader",
    "aggressive_risk",
    "neutral_risk",
    "conservative_risk",
    "risk_debate_consolidation",
    "portfolio_manager",
]

ChatModelOverride = Callable[[CallSite | None], BaseChatModel]

_chat_model_override: ChatModelOverride | None = None


def override_chat_models(override: ChatModelOverride | None) -> ChatModelOverride | None:
    global _chat_model_override  # noqa: PLW0603
    previous = _chat_model_override
    _chat_model_override = override
    return previous


def create_llm_client(
    tier: str,
    config: Settings,
    temperature: float | None = None,
    call_site: CallSite | None = None,
) -> BaseChatModel:
    if _chat_model_override is not None:
        return _chat_model_override(call_site)

    if tier not in ("deep_think", "quick_think"):
        raise ValueError(f"Invalid tier '{tier}'. Must be 'deep_think' or 'quick_think'.")

    model_name = resolve_model(
        config, config.llm_provider, "deep" if tier == "deep_think" else "quick"
    )
    provider = config.llm_provider

    if provider == "openrouter":
        from finagents.llm_clients.openrouter_client import (  # noqa: PLC0415
            create_openrouter_client,
        )

        return create_openrouter_client(
            model=model_name,
            api_key=config.openrouter_api_key,
            temperature=temperature,
        )

    elif provider == "cerebras":
        from finagents.llm_clients.cerebras_client import create_cerebras_client  # noqa: PLC0415

        return create_cerebras_client(
            model=model_name,
            api_key=config.cerebras_api_key,
            temperature=temperature,
        )

    elif provider == "groq":
        from finagents.llm_clients.groq_client import create_groq_client  # noqa: PLC0415

        return create_groq_client(
            model=model_name,
            api_key=config.groq_api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            "Valid options: 'openrouter', 'cerebras', 'groq'"
        )
