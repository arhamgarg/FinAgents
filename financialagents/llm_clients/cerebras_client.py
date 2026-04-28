from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_cerebras_client(
    model: str,
    api_key: str | None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is required to use the Cerebras provider. "
            "Set it in your.env file or environment."
        )
    from langchain_cerebras import ChatCerebras

    kwargs: dict[str, object] = {"model": model, "api_key": api_key}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatCerebras(**kwargs)  # ty: ignore[invalid-argument-type]
