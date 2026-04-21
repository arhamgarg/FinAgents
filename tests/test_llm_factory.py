from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from langchain_core.language_models import BaseChatModel

from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


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
            mock_ollama.assert_called_once_with(model=default_settings.quick_think_llm)


class TestGoogleProviderDropped:

    def test_google_provider_rejected(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with pytest.raises(ValidationError, match="llm_provider"):
            Settings(_env_file=None)


class TestMaxOutputTokensPlumbing:

    def test_groq_max_tokens_kwarg_translation(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", groq_settings, max_output_tokens=1024)
            _, kwargs = mock_groq.call_args
            assert kwargs.get("max_tokens") == 1024, f"Expected max_tokens=1024, got {kwargs}"

    def test_ollama_num_predict_kwarg_translation(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", default_settings, max_output_tokens=1024)
            _, kwargs = mock_ollama.call_args
            assert kwargs.get("num_predict") == 1024, f"Expected num_predict=1024, got {kwargs}"
            assert "max_tokens" not in kwargs, (
                f"P18 regression: ChatOllama must NOT receive max_tokens (silently ignored). kwargs={kwargs}"
            )

    def test_no_token_arg_keeps_backward_compat_groq(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", groq_settings)
            _, kwargs = mock_groq.call_args
            assert "max_tokens" not in kwargs or kwargs["max_tokens"] is None

    def test_no_token_arg_keeps_backward_compat_ollama(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", default_settings)
            _, kwargs = mock_ollama.call_args
            assert "num_predict" not in kwargs or kwargs["num_predict"] is None

    def test_max_output_tokens_is_keyword_only(self) -> None:
        import inspect

        sig = inspect.signature(create_llm_client)
        assert "max_output_tokens" in sig.parameters
        assert sig.parameters["max_output_tokens"].default is None


class TestFactoryTemperaturePassthrough:

    def test_groq_receives_temperature_when_provided(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client(
                "quick_think", groq_settings, max_output_tokens=1024, temperature=0.9
            )
            _, kwargs = mock_groq.call_args
            assert kwargs.get("temperature") == 0.9, (
                f"Expected temperature=0.9, got {kwargs}"
            )
            assert kwargs.get("max_tokens") == 1024, f"Expected max_tokens=1024, got {kwargs}"

    def test_groq_omits_temperature_when_not_provided(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", groq_settings, max_output_tokens=1024)
            _, kwargs = mock_groq.call_args
            assert "temperature" not in kwargs, (
                f"Backward-compat regression: temperature should not be passed when None; "
                f"kwargs={kwargs}"
            )

    def test_ollama_receives_temperature_when_provided(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client(
                "quick_think", default_settings, max_output_tokens=1024, temperature=0.3
            )
            _, kwargs = mock_ollama.call_args
            assert kwargs.get("temperature") == 0.3, (
                f"Expected temperature=0.3, got {kwargs}"
            )
            assert kwargs.get("num_predict") == 1024, (
                f"Expected num_predict=1024, got {kwargs}"
            )

    def test_ollama_omits_temperature_when_not_provided(self, default_settings: Settings) -> None:
        with patch("langchain_ollama.ChatOllama", create=True) as mock_ollama:
            mock_ollama.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", default_settings, max_output_tokens=1024)
            _, kwargs = mock_ollama.call_args
            assert "temperature" not in kwargs, (
                f"Backward-compat regression: temperature should not be passed when None; "
                f"kwargs={kwargs}"
            )

    def test_temperature_kwarg_signature(self) -> None:
        import inspect

        sig = inspect.signature(create_llm_client)
        assert "temperature" in sig.parameters
        assert sig.parameters["temperature"].default is None
