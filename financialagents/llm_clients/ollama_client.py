from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_ollama_client(model: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model)
