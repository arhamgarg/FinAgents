from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client
from financialagents.llm_clients.model_catalog import (
    GROQ_MODELS,
    OLLAMA_MODELS,
    get_default_model,
)


class TestFactoryReturnsBaseChatModel:

    def test_ollama_returns_base_chat_model(self, default_settings: Settings) -> None:
        with patch("financialagents.llm_clients.ollama_client.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", default_settings)
            assert isinstance(llm, BaseChatModel)

    def test_groq_returns_base_chat_model(self, groq_settings: Settings) -> None:
        with patch("financialagents.llm_clients.groq_client.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", groq_settings)
            assert isinstance(llm, BaseChatModel)


class TestTierRouting:

    def test_deep_think_uses_deep_model(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", default_settings)
            mock_ollama.assert_called_once_with(model=default_settings.deep_think_llm)

    def test_quick_think_uses_quick_model(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", default_settings)
            mock_ollama.assert_called_once_with(model=default_settings.quick_think_llm)

    def test_groq_tier_routing(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", groq_settings)
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args
            assert call_kwargs.kwargs.get("model") == groq_settings.deep_think_llm

    def test_invalid_tier_raises(self, default_settings: Settings) -> None:
        with pytest.raises(ValueError, match="Invalid tier"):
            create_llm_client("medium_think", default_settings)


class TestGroqProvider:

    def test_groq_missing_key_raises(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        config = Settings(_env_file=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            create_llm_client("deep_think", config)

    def test_groq_with_key_succeeds(self, groq_settings: Settings) -> None:
        with patch("financialagents.llm_clients.groq_client.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", groq_settings)
            assert isinstance(llm, BaseChatModel)


class TestOllamaProvider:

    def test_ollama_no_key_required(self, default_settings: Settings) -> None:
        with patch("financialagents.llm_clients.ollama_client.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", default_settings)
            assert isinstance(llm, BaseChatModel)

    def test_ollama_uses_correct_model(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", default_settings)
            mock_ollama.assert_called_once_with(model="gemma4:latest")


class TestGoogleProviderDropped:

    def test_google_provider_rejected(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with pytest.raises(Exception):
            Settings(_env_file=None)


class TestModelCatalog:

    def test_groq_deep_think_default(self) -> None:
        assert get_default_model("groq", "deep_think") == "llama-3.3-70b-versatile"

    def test_groq_quick_think_default(self) -> None:
        assert get_default_model("groq", "quick_think") == "llama-3.1-8b-instant"

    def test_ollama_deep_think_default(self) -> None:
        assert get_default_model("ollama", "deep_think") == "gemma4:latest"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_default_model("openai", "deep_think")

    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tier"):
            get_default_model("groq", "turbo_think")
