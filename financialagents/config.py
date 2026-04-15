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

    angelone_api_key: Optional[str] = None
    angelone_client_id: Optional[str] = None
    angelone_pin: Optional[str] = None
    angelone_totp_secret: Optional[str] = None

    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "financialagents/1.0"

    stocktwits_token: Optional[str] = None

    data_vendors: dict[str, str] = Field(
        default_factory=lambda: {
            "ohlcv": "angelone",
            "fundamentals": "screener",
            "news": "moneycontrol",
            "sentiment": "reddit",
            "sectoral": "nse",
        }
    )

    analyst_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 1.0,
            "fundamentals": 1.5,
            "news": 0.8,
            "sentiment": 0.7,
        }
    )

    max_debate_rounds: int = 3
    max_risk_discuss_rounds: int = 2

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
        if not self.reddit_client_id or not self.reddit_client_secret:
            warnings.warn(
                "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set. Reddit sentiment will return DataResult(status='empty') at use time.",
                UserWarning,
                stacklevel=2,
            )
        return self
