from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_groq_client(
    model: str,
    api_key: str | None,
    temperature: float | None = None,
) -> BaseChatModel:
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required to use the Groq provider. "
            "Set it in your .env file or environment."
        )
    from langchain_groq import ChatGroq  # noqa: PLC0415

    kwargs: dict[str, object] = {"model": model, "api_key": api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGroq(**kwargs)  # ty: ignore[invalid-argument-type]
