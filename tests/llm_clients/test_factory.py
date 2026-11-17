from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from finagents.config import Settings, resolve_model
from finagents.llm_clients.factory import create_llm_client


class TestFactoryReturnsBaseChatModel:
    def test_groq_returns_base_chat_model(self, groq_settings: Settings) -> None:
        with patch("finagents.llm_clients.groq_client.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", groq_settings)
            assert isinstance(llm, BaseChatModel)


class TestTierRouting:
    def test_deep_think_uses_deep_model(self, default_settings: Settings) -> None:
        with patch("langchain_openai.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", default_settings)
            mock_or.assert_called_once()
            _, kwargs = mock_or.call_args
            assert kwargs.get("model") == resolve_model(default_settings, "openrouter", "deep")

    def test_quick_think_uses_quick_model(self, default_settings: Settings) -> None:
        with patch("langchain_openai.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", default_settings)
            mock_or.assert_called_once()
            _, kwargs = mock_or.call_args
            assert kwargs.get("model") == resolve_model(default_settings, "openrouter", "quick")

    def test_groq_tier_routing(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", groq_settings)
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args
            assert call_kwargs.kwargs.get("model") == resolve_model(groq_settings, "groq", "deep")

    def test_invalid_tier_raises(self, default_settings: Settings) -> None:
        with pytest.raises(ValueError, match="Invalid tier"):
            create_llm_client("medium_think", default_settings)


class TestGroqProvider:
    def test_groq_missing_key_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        config = Settings()
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            create_llm_client("deep_think", config)

    def test_groq_with_key_succeeds(self, groq_settings: Settings) -> None:
        with patch("finagents.llm_clients.groq_client.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", groq_settings)
            assert isinstance(llm, BaseChatModel)


class TestOpenRouterProvider:
    def test_openrouter_missing_key_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("GROQ_API_KEY", "fake-scaffold")
        config = Settings()
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_llm_client("deep_think", config)

    def test_openrouter_with_key_succeeds(self, openrouter_settings: Settings) -> None:
        with patch("finagents.llm_clients.openrouter_client.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", openrouter_settings)
            assert isinstance(llm, BaseChatModel)

    def test_openrouter_passes_base_url(self, openrouter_settings: Settings) -> None:
        with patch("langchain_openai.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", openrouter_settings)
            _, kwargs = mock_or.call_args
            assert kwargs.get("base_url") == "https://openrouter.ai/api/v1"

    def test_openrouter_tier_routing(self, openrouter_settings: Settings) -> None:
        with patch("langchain_openai.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", openrouter_settings)
            _, kwargs = mock_or.call_args
            assert kwargs.get("model") == resolve_model(openrouter_settings, "openrouter", "quick")

    def test_openrouter_temperature_kwarg(self, openrouter_settings: Settings) -> None:
        with patch("langchain_openai.ChatOpenAI", create=True) as mock_or:
            mock_or.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", openrouter_settings, temperature=0.7)
            _, kwargs = mock_or.call_args
            assert kwargs.get("temperature") == 0.7


class TestCerebrasProvider:
    def test_cerebras_missing_key_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "cerebras")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        config = Settings()
        with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
            create_llm_client("deep_think", config)

    def test_cerebras_with_key_succeeds(self, cerebras_settings: Settings) -> None:
        with patch("finagents.llm_clients.cerebras_client.ChatCerebras", create=True) as mock_cb:
            mock_cb.return_value = MagicMock(spec=BaseChatModel)
            llm = create_llm_client("deep_think", cerebras_settings)
            assert isinstance(llm, BaseChatModel)

    def test_cerebras_tier_routing(self, cerebras_settings: Settings) -> None:
        with patch("langchain_cerebras.ChatCerebras", create=True) as mock_cb:
            mock_cb.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", cerebras_settings)
            _, kwargs = mock_cb.call_args
            assert kwargs.get("model") == resolve_model(cerebras_settings, "cerebras", "quick")

    def test_cerebras_temperature_kwarg(self, cerebras_settings: Settings) -> None:
        with patch("langchain_cerebras.ChatCerebras", create=True) as mock_cb:
            mock_cb.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("deep_think", cerebras_settings, temperature=0.7)
            _, kwargs = mock_cb.call_args
            assert kwargs.get("temperature") == 0.7

    def test_cerebras_wso_json_mode_default(self) -> None:
        from finagents.llm_clients.cerebras_client import _apply_json_mode_default

        mock_llm = MagicMock(spec=BaseChatModel)
        original_wso = MagicMock(return_value=MagicMock())
        mock_llm.with_structured_output = original_wso

        from pydantic import BaseModel as PydanticBaseModel

        class _Dummy(PydanticBaseModel):
            items: list[str]

        patched = _apply_json_mode_default(mock_llm)

        patched.with_structured_output(_Dummy)
        original_wso.assert_called_once_with(_Dummy, method="json_mode", include_raw=False)

    def test_cerebras_wso_explicit_method_passthrough(self) -> None:
        from finagents.llm_clients.cerebras_client import _apply_json_mode_default

        mock_llm = MagicMock(spec=BaseChatModel)
        original_wso = MagicMock(return_value=MagicMock())
        mock_llm.with_structured_output = original_wso

        from pydantic import BaseModel as PydanticBaseModel

        class _Dummy(PydanticBaseModel):
            value: str

        patched = _apply_json_mode_default(mock_llm)
        patched.with_structured_output(_Dummy, method="function_calling")

        original_wso.assert_called_once_with(_Dummy, method="function_calling", include_raw=False)


class TestGoogleProviderDropped:
    def test_google_provider_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with pytest.raises(ValidationError, match="llm_provider"):
            Settings()


class TestFactoryTemperaturePassthrough:
    def test_groq_receives_temperature_when_provided(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", groq_settings, temperature=0.9)
            _, kwargs = mock_groq.call_args
            assert kwargs.get("temperature") == 0.9, f"Expected temperature=0.9, got {kwargs}"

    def test_groq_omits_temperature_when_not_provided(self, groq_settings: Settings) -> None:
        with patch("langchain_groq.ChatGroq", create=True) as mock_groq:
            mock_groq.return_value = MagicMock(spec=BaseChatModel)
            create_llm_client("quick_think", groq_settings)
            _, kwargs = mock_groq.call_args
            assert "temperature" not in kwargs, (
                f"Backward-compat regression: temperature should not be passed when None; "
                f"kwargs={kwargs}"
            )

    def test_temperature_kwarg_signature(self) -> None:
        import inspect

        sig = inspect.signature(create_llm_client)
        assert "temperature" in sig.parameters
        assert sig.parameters["temperature"].default is None
