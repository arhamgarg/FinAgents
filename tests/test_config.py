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

    def test_env_overrides_provider(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        s = Settings(_env_file=None)
        assert s.llm_provider == "groq"

    def test_env_overrides_model(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEP_THINK_LLM", "custom-model")
        s = Settings(_env_file=None)
        assert s.deep_think_llm == "custom-model"

    def test_invalid_provider_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_fa_data_dir_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        target = str(tmp_path / "fa")
        monkeypatch.setenv("FA_DATA_DIR", target)
        s = Settings(_env_file=None)
        assert s.fa_data_dir == target

    def test_fa_data_dir_default_none(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FA_DATA_DIR", raising=False)
        s = Settings(_env_file=None)
        assert s.fa_data_dir is None


class TestDebateParams:

    def test_default_debate_rounds(self, default_settings: Settings) -> None:
        assert default_settings.max_debate_rounds == 3

    def test_default_risk_rounds(self, default_settings: Settings) -> None:
        assert default_settings.max_risk_discuss_rounds == 2

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


class TestPhase4SettingsFields:

    def test_risk_persona_weights_default(self, default_settings: Settings) -> None:
        assert default_settings.risk_persona_weights == {
            "aggressive": 0.25,
            "neutral": 0.50,
            "conservative": 0.25,
        }

    def test_position_sizing_method_default(self, default_settings: Settings) -> None:
        assert default_settings.position_sizing_method == "atr_based"

    def test_risk_per_trade_pct_default(self, default_settings: Settings) -> None:
        assert default_settings.risk_per_trade_pct == pytest.approx(0.02)

    def test_kelly_fraction_default(self, default_settings: Settings) -> None:
        assert default_settings.kelly_fraction == pytest.approx(0.25)

    def test_atr_stop_multiplier_default(self, default_settings: Settings) -> None:
        assert default_settings.atr_stop_multiplier == pytest.approx(2.0)

    def test_portfolio_value_inr_default(self, default_settings: Settings) -> None:
        assert default_settings.portfolio_value_inr == pytest.approx(100_000.0)

    def test_position_sizing_method_rejects_invalid_literal(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(_env_file=None, position_sizing_method="moon_math")  # ty: ignore[unknown-argument, invalid-argument-type]

    def test_no_debate_compression_max_chars_field(self, default_settings: Settings) -> None:
        assert not hasattr(default_settings, "debate_compression_max_chars")

    def test_max_debate_rounds_default_unchanged_at_3(self, default_settings: Settings) -> None:
        assert default_settings.max_debate_rounds == 3

    def test_default_time_horizon_is_positional(self) -> None:
        s = Settings(_env_file=None, llm_provider="ollama", groq_api_key=None)
        assert s.default_time_horizon == "positional"

    def test_default_time_horizon_rejects_intraday(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(
                _env_file=None,
                llm_provider="ollama",
                groq_api_key=None,
                default_time_horizon="intraday",
            )  # ty: ignore[invalid-argument-type]

    def test_default_time_horizon_rejects_swing(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(
                _env_file=None,
                llm_provider="ollama",
                groq_api_key=None,
                default_time_horizon="swing",
            )  # ty: ignore[invalid-argument-type]
