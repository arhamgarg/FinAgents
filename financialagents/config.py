from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    llm_provider: Literal["groq", "ollama"] = "groq"
    deep_think_llm: str = "openai/gpt-oss-120b"
    quick_think_llm: str = "openai/gpt-oss-20b"

    groq_api_key: Optional[str] = None
    ollama_api_key: Optional[str] = None

    angelone_api_key: Optional[str] = None
    angelone_client_id: Optional[str] = None
    angelone_pin: Optional[str] = None
    angelone_totp_secret: Optional[str] = None

    data_vendors: dict[str, str] = Field(
        default_factory=lambda: {
            "ohlcv": "angelone",
            "fundamentals": "screener",
            "news": "moneycontrol",
            "sectoral": "nse",
        }
    )

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

    sector_rotation_enabled: bool = True
    max_sectors: int = 3
    stocks_per_sector: int = 5

    output_language: str = "en"

    @model_validator(mode="after")
    def warn_missing_optional_keys(self) -> Settings:
        if self.llm_provider == "groq" and not self.groq_api_key:
            warnings.warn(
                "GROQ_API_KEY is not set. Groq provider calls will fail at use time.",
                UserWarning,
                stacklevel=2,
            )
        if not self.angelone_api_key:
            warnings.warn(
                "ANGELONE_API_KEY is not set. OHLCV and fundamentals fallback will return DataResult(status='empty') at use time.",
                UserWarning,
                stacklevel=2,
            )
        return self
