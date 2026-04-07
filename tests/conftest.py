from __future__ import annotations

import os
from typing import Generator

import pytest

from financialagents.config import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "LLM_PROVIDER",
        "DEEP_THINK_LLM",
        "QUICK_THINK_LLM",
        "GROQ_API_KEY",
        "ANGELONE_API_KEY",
        "ANGELONE_CLIENT_ID",
        "ANGELONE_PIN",
        "ANGELONE_TOTP_SECRET",
        "MAX_DEBATE_ROUNDS",
        "MAX_RISK_DISCUSS_ROUNDS",
        "MAX_SECTORS",
        "STOCKS_PER_SECTOR",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def default_settings(clean_env: None) -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def groq_settings(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key-fake")
    return Settings(_env_file=None)
