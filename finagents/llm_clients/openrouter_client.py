from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_openrouter_client(
    model: str,
    api_key: str | None,
    temperature: float | None = None,
) -> BaseChatModel:
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required to use the OpenRouter provider. "
            "Set it in your.env file or environment."
        )
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)  # ty: ignore[invalid-argument-type]
