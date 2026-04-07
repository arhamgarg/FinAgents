from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    llm_provider: Literal["groq", "ollama"] = "ollama"
    deep_think_llm: str = "gemma4:latest"
    quick_think_llm: str = "gemma4:latest"

    groq_api_key: Optional[str] = None

    angelone_api_key: Optional[str] = None
    angelone_client_id: Optional[str] = None
    angelone_pin: Optional[str] = None
    angelone_totp_secret: Optional[str] = None

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
        return self
