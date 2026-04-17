from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_ollama_client(
    model: str,
    num_predict: int | None = None,
) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    kwargs: dict[str, object] = {"model": model}
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    return ChatOllama(**kwargs)  # ty: ignore[unknown-argument, invalid-argument-type]
