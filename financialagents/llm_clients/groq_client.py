from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_groq_client(model: str, api_key: str | None) -> BaseChatModel:
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required to use the Groq provider. "
            "Set it in your.env file or environment."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, api_key=api_key)  # ty: ignore[unknown-argument]
