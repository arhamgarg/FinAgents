from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _hermetic_provider_guard(request):
    if any(
        m.name == "allow_provider_clients" for m in request.node.iter_markers()
    ):
        yield
        return

    import langchain_cerebras
    import langchain_groq
    import langchain_ollama
    import langchain_openai

    msg = "provider client instantiated under hermetic test"
    with (
        patch.object(
            langchain_groq.ChatGroq, "__init__", side_effect=RuntimeError(msg)
        ),
        patch.object(
            langchain_ollama.ChatOllama, "__init__", side_effect=RuntimeError(msg)
        ),
        patch.object(
            langchain_openai.ChatOpenAI, "__init__", side_effect=RuntimeError(msg)
        ),
        patch.object(
            langchain_cerebras.ChatCerebras, "__init__", side_effect=RuntimeError(msg)
        ),
    ):
        yield
