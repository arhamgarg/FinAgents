from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_ollama_client(
    model: str,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    kwargs: dict[str, object] = {"model": model}
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOllama(**kwargs)  # ty: ignore[invalid-argument-type]
