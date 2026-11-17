from __future__ import annotations

import warnings
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openrouter": {"deep": "openrouter/auto", "quick": "openrouter/auto"},
    "cerebras": {"deep": "gpt-oss-120b", "quick": "llama3.1-8b"},
    "groq": {"deep": "openai/gpt-oss-120b", "quick": "openai/gpt-oss-20b"},
}


def resolve_model(
    settings: Settings,
    provider: str,
    tier: Literal["deep", "quick"],
) -> str:
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider {provider!r}. Valid: {sorted(PROVIDER_DEFAULTS)}")
    override = settings.deep_think_llm if tier == "deep" else settings.quick_think_llm
    return override if override is not None else PROVIDER_DEFAULTS[provider][tier]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    llm_provider: Literal["openrouter", "cerebras", "groq"] = "openrouter"
    deep_think_llm: str | None = None
    quick_think_llm: str | None = None

    openrouter_api_key: str | None = None
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None

    fa_data_dir: str | None = None

    angelone_api_key: str | None = None
    angelone_client_id: str | None = None
    angelone_pin: str | None = None
    angelone_totp_secret: str | None = None

    analyst_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 0.303,
            "fundamentals": 0.455,
            "news": 0.242,
        }
    )

    max_debate_rounds: int = 3
    max_risk_discuss_rounds: int = 2

    risk_persona_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "aggressive": 0.25,
            "neutral": 0.50,
            "conservative": 0.25,
        }
    )

    position_sizing_method: Literal["atr_based", "kelly", "fixed_fractional"] = "atr_based"
    risk_per_trade_pct: float = 0.02
    kelly_fraction: float = 0.25
    atr_stop_multiplier: float = 2.0
    portfolio_value_inr: float = 100_000.0

    default_time_horizon: Literal["positional"] = "positional"

    screener_min_delay: float = 4.0
    moneycontrol_min_delay: float = 2.0
    economictimes_min_delay: float = 2.0
    livemint_min_delay: float = 2.0
    nse_min_delay: float = 1.0
    rate_jitter_max: float = 2.0

    @model_validator(mode="after")
    def warn_missing_optional_keys(self) -> Settings:
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            warnings.warn(
                "OPENROUTER_API_KEY is not set. OpenRouter provider calls will fail at use time.",
                UserWarning,
                stacklevel=2,
            )
        if self.llm_provider == "cerebras" and not self.cerebras_api_key:
            warnings.warn(
                "CEREBRAS_API_KEY is not set. Cerebras provider calls will fail at use time.",
                UserWarning,
                stacklevel=2,
            )
        if self.llm_provider == "groq" and not self.groq_api_key:
            warnings.warn(
                "GROQ_API_KEY is not set. Groq provider calls will fail at use time.",
                UserWarning,
                stacklevel=2,
            )
        if not self.angelone_api_key:
            warnings.warn(
                "ANGELONE_API_KEY is not set. OHLCV and fundamentals fallback will return "
                "DataResult(status='empty') at use time.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def require_at_least_one_llm_key(self) -> Settings:
        if not (self.openrouter_api_key or self.cerebras_api_key or self.groq_api_key):
            raise ValueError(
                "No LLM provider API key configured. Set at least one of: "
                "OPENROUTER_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY. "
                "See README for setup."
            )
        return self
