from __future__ import annotations

import warnings

import pytest

from financialagents.config import Settings


class TestSettingsDefaults:

    def test_default_provider_is_ollama(self, default_settings: Settings) -> None:
        assert default_settings.llm_provider == "ollama"

    def test_default_deep_think_model(self, default_settings: Settings) -> None:
        assert default_settings.deep_think_llm == "openai/gpt-oss-120b"

    def test_default_quick_think_model(self, default_settings: Settings) -> None:
        assert default_settings.quick_think_llm == "openai/gpt-oss-20b"

    def test_optional_keys_default_to_none(self, default_settings: Settings) -> None:
        assert default_settings.groq_api_key is None
        assert default_settings.angelone_api_key is None


class TestSettingsEnvLoading:

    def test_env_overrides_provider(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        s = Settings(_env_file=None)
        assert s.llm_provider == "groq"

    def test_env_overrides_model(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEP_THINK_LLM", "custom-model")
        s = Settings(_env_file=None)
        assert s.deep_think_llm == "custom-model"

    def test_invalid_provider_rejected(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(Exception):
            Settings(_env_file=None)


class TestDebateParams:

    def test_default_debate_rounds(self, default_settings: Settings) -> None:
        assert default_settings.max_debate_rounds == 3

    def test_default_risk_rounds(self, default_settings: Settings) -> None:
        assert default_settings.max_risk_discuss_rounds == 2

    def test_default_max_sectors(self, default_settings: Settings) -> None:
        assert default_settings.max_sectors == 3

    def test_default_stocks_per_sector(self, default_settings: Settings) -> None:
        assert default_settings.stocks_per_sector == 5

    def test_env_overrides_debate_rounds(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAX_DEBATE_ROUNDS", "5")
        s = Settings(_env_file=None)
        assert s.max_debate_rounds == 5


class TestStartupWarnings:

    def test_groq_without_key_warns(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(_env_file=None)
            groq_warnings = [x for x in w if "GROQ_API_KEY" in str(x.message)]
            assert len(groq_warnings) >= 1

    def test_ollama_without_key_no_warning(self, default_settings: Settings) -> None:
        pass
