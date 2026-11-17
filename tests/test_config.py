from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from finagents.config import PROVIDER_DEFAULTS, Settings, resolve_model


class TestSettingsDefaults:
    def test_default_provider_is_openrouter(self, default_settings: Settings) -> None:
        assert default_settings.llm_provider == "openrouter"

    def test_default_deep_think_model_is_none(self, default_settings: Settings) -> None:
        assert default_settings.deep_think_llm is None

    def test_default_quick_think_model_is_none(self, default_settings: Settings) -> None:
        assert default_settings.quick_think_llm is None

    def test_optional_keys_default_to_none(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(openrouter_api_key="fake")
        assert s.groq_api_key is None
        assert s.cerebras_api_key is None
        assert s.angelone_api_key is None

    def test_schema_default_provider_is_openrouter(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(openrouter_api_key="fake")
        assert s.llm_provider == "openrouter"


class TestSettingsEnvLoading:
    def test_env_overrides_provider(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        s = Settings()
        assert s.llm_provider == "groq"

    def test_env_overrides_model(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEP_THINK_LLM", "custom-model")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        s = Settings()
        assert s.deep_think_llm == "custom-model"

    def test_invalid_provider_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(ValidationError):
            Settings()

    def test_fa_data_dir_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        target = str(tmp_path / "fa")
        monkeypatch.setenv("FA_DATA_DIR", target)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        s = Settings()
        assert s.fa_data_dir == target

    def test_fa_data_dir_default_none(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FA_DATA_DIR", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        s = Settings()
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
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        s = Settings()
        assert s.max_debate_rounds == 5


class TestStartupWarnings:
    def test_groq_without_key_warns(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings()
            groq_warnings = [x for x in w if "GROQ_API_KEY" in str(x.message)]
            assert len(groq_warnings) >= 1

    def test_openrouter_without_key_warns(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("GROQ_API_KEY", "fake-scaffold")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings()
            hits = [x for x in w if "OPENROUTER_API_KEY" in str(x.message)]
            assert len(hits) >= 1

    def test_cerebras_without_key_warns(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "cerebras")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-scaffold")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings()
            hits = [x for x in w if "CEREBRAS_API_KEY" in str(x.message)]
            assert len(hits) >= 1


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
            Settings(position_sizing_method="moon_math")  # ty: ignore[invalid-argument-type]

    def test_no_debate_compression_max_chars_field(self, default_settings: Settings) -> None:
        assert not hasattr(default_settings, "debate_compression_max_chars")

    def test_max_debate_rounds_default_unchanged_at_3(self, default_settings: Settings) -> None:
        assert default_settings.max_debate_rounds == 3

    def test_default_time_horizon_is_positional(self) -> None:
        s = Settings(llm_provider="openrouter", openrouter_api_key="fake")
        assert s.default_time_horizon == "positional"

    def test_default_time_horizon_rejects_intraday(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(
                llm_provider="openrouter",
                openrouter_api_key="fake",
                default_time_horizon="intraday",  # ty: ignore[invalid-argument-type]
            )

    def test_default_time_horizon_rejects_swing(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(
                llm_provider="openrouter",
                openrouter_api_key="fake",
                default_time_horizon="swing",  # ty: ignore[invalid-argument-type]
            )


class TestResolveModel:
    def test_override_wins_for_deep_tier(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(
                deep_think_llm="anthropic/claude-3.5-sonnet",
                openrouter_api_key="fake",
            )

        assert resolve_model(s, "groq", "deep") == "anthropic/claude-3.5-sonnet"
        assert resolve_model(s, "cerebras", "deep") == "anthropic/claude-3.5-sonnet"
        assert resolve_model(s, "openrouter", "deep") == "anthropic/claude-3.5-sonnet"

    def test_override_wins_for_quick_tier(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(quick_think_llm="custom-quick-model", openrouter_api_key="fake")

        assert resolve_model(s, "groq", "quick") == "custom-quick-model"
        assert resolve_model(s, "openrouter", "quick") == "custom-quick-model"
        assert resolve_model(s, "cerebras", "quick") == "custom-quick-model"

    def test_provider_defaults_keys_match_llm_provider_literal(self) -> None:
        assert set(PROVIDER_DEFAULTS.keys()) == {"openrouter", "cerebras", "groq"}
        for entry in PROVIDER_DEFAULTS.values():
            assert set(entry.keys()) == {"deep", "quick"}
            for v in entry.values():
                assert isinstance(v, str) and v != ""

    def test_default_resolves_to_openrouter_default(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(deep_think_llm=None, quick_think_llm=None, openrouter_api_key="fake")

        assert resolve_model(s, "openrouter", "deep") == PROVIDER_DEFAULTS["openrouter"]["deep"]
        assert resolve_model(s, "openrouter", "quick") == PROVIDER_DEFAULTS["openrouter"]["quick"]

    def test_default_resolves_per_provider(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(deep_think_llm=None, quick_think_llm=None, openrouter_api_key="fake")

        assert resolve_model(s, "cerebras", "deep") == PROVIDER_DEFAULTS["cerebras"]["deep"]
        assert resolve_model(s, "cerebras", "quick") == PROVIDER_DEFAULTS["cerebras"]["quick"]
        assert resolve_model(s, "groq", "deep") == PROVIDER_DEFAULTS["groq"]["deep"]
        assert resolve_model(s, "groq", "quick") == PROVIDER_DEFAULTS["groq"]["quick"]

    def test_override_one_tier_other_tier_falls_through(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s_deep = Settings(
                deep_think_llm="anthropic/claude-3.5-sonnet",
                quick_think_llm=None,
                openrouter_api_key="fake",
            )

        assert resolve_model(s_deep, "groq", "deep") == "anthropic/claude-3.5-sonnet"
        assert resolve_model(s_deep, "groq", "quick") == PROVIDER_DEFAULTS["groq"]["quick"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s_quick = Settings(
                deep_think_llm=None,
                quick_think_llm="custom-quick-model",
                openrouter_api_key="fake",
            )

        assert resolve_model(s_quick, "groq", "quick") == "custom-quick-model"
        assert resolve_model(s_quick, "groq", "deep") == PROVIDER_DEFAULTS["groq"]["deep"]

    def test_unknown_provider_raises_value_error(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings(openrouter_api_key="fake")

        with pytest.raises(ValueError, match=r"Unsupported provider 'grokk'"):
            resolve_model(s, "grokk", "deep")


class TestRequireAtLeastOneLlmKey:
    def test_zero_keys_raises_value_error(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finagents.config import Settings

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with pytest.raises(ValueError, match=r"No LLM provider API key"):
            Settings()

    def test_error_message_lists_three_env_vars(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finagents.config import Settings

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with pytest.raises(ValueError) as exc_info:
            Settings()
        msg = str(exc_info.value)
        assert "OPENROUTER_API_KEY" in msg
        assert "CEREBRAS_API_KEY" in msg
        assert "GROQ_API_KEY" in msg

    def test_one_key_is_enough(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        from finagents.config import Settings

        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings()
        assert s.openrouter_api_key == "fake"

    def test_multiple_keys_no_error(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        from finagents.config import Settings

        monkeypatch.setenv("OPENROUTER_API_KEY", "fake1")
        monkeypatch.setenv("GROQ_API_KEY", "fake2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings()
        assert s.openrouter_api_key == "fake1"
        assert s.groq_api_key == "fake2"

    def test_ollama_provider_rejected(self, clean_env: None) -> None:
        import pydantic

        from finagents.config import Settings

        with pytest.raises(pydantic.ValidationError, match="llm_provider"):
            Settings(llm_provider="ollama", openrouter_api_key="fake")  # ty: ignore[invalid-argument-type]

    def test_stale_ollama_api_key_ignored(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_API_KEY", "leftover-value")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s = Settings()
        assert s.openrouter_api_key == "fake"
        assert not hasattr(s, "ollama_api_key")


class TestSettingsAnalystWeights:
    def test_analyst_weights_default_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from finagents.config import Settings

            s = Settings(openrouter_api_key="fake")
        assert s.analyst_weights == {
            "technical": 0.303,
            "fundamentals": 0.455,
            "news": 0.242,
        }
        assert abs(sum(s.analyst_weights.values()) - 1.0) < 1e-9
