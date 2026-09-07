from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.config import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    from finagents.dataflows import cache
    from tests._network import block_network

    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field.lower(), raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-api-key-fake")
    monkeypatch.setenv("FA_DATA_DIR", str(tmp_path / "signals"))
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "market.db")
    with block_network():
        yield


@pytest.fixture
def hermetic_provider_guard():
    import langchain_cerebras
    import langchain_groq
    import langchain_openai

    msg = "provider client instantiated under hermetic test"
    with (
        patch.object(langchain_groq.ChatGroq, "__init__", side_effect=RuntimeError(msg)),
        patch.object(langchain_openai.ChatOpenAI, "__init__", side_effect=RuntimeError(msg)),
        patch.object(langchain_cerebras.ChatCerebras, "__init__", side_effect=RuntimeError(msg)),
    ):
        yield


@pytest.fixture
def stub_surveillance(monkeypatch):
    monkeypatch.setattr(
        "finagents.agents.analysts.technical_analyst.get_surveillance_data",
        AsyncMock(return_value=(0.20, False)),
    )


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "LLM_PROVIDER",
        "DEEP_THINK_LLM",
        "QUICK_THINK_LLM",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
        "ANGELONE_API_KEY",
        "ANGELONE_CLIENT_ID",
        "ANGELONE_PIN",
        "ANGELONE_TOTP_SECRET",
        "MAX_DEBATE_ROUNDS",
        "MAX_RISK_DISCUSS_ROUNDS",
        "FA_DATA_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def default_settings(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-api-key-fake")
    return Settings()


@pytest.fixture
def groq_settings(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key-fake")
    return Settings()


@pytest.fixture
def openrouter_settings(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-api-key-fake")
    return Settings()


@pytest.fixture
def cerebras_settings(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-api-key-fake")
    return Settings()


@pytest.fixture
def recorded_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_smart_connect() -> Generator[MagicMock, None, None]:
    with patch("finagents.dataflows.session_manager.SmartConnect") as mock_cls:
        instance = MagicMock()
        instance.generateSession.return_value = {
            "status": True,
            "message": "SUCCESS",
            "data": {
                "jwtToken": "test-jwt-token",
                "refreshToken": "test-refresh-token",
                "feedToken": "test-feed-token",
            },
        }
        instance.getCandleData.return_value = {
            "status": True,
            "data": [
                ["2024-01-01T09:15:00+05:30", 2845.5, 2900.0, 2840.0, 2880.0, 100000],
            ],
        }
        instance.getProfile.return_value = {"status": True, "data": {"clientcode": "TEST01"}}
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_toolkit(
    mock_smart_connect: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    monkeypatch.setenv("ANGELONE_API_KEY", "test-api-key")
    monkeypatch.setenv("ANGELONE_CLIENT_ID", "TEST01")
    monkeypatch.setenv("ANGELONE_PIN", "1234")
    monkeypatch.setenv("ANGELONE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    test_cache = tmp_path / "test_cache.db"
    monkeypatch.setattr(
        "finagents.dataflows.cache.CACHE_DB_PATH",
        test_cache,
    )
    toolkit = MagicMock()
    toolkit.smart_connect = mock_smart_connect
    toolkit.cache_path = test_cache
    toolkit.fixture_dir = Path(__file__).parent / "fixtures"
    return toolkit


@pytest.fixture
def llm_harness():
    from finagents.testing.llm_mock import LLMMockHarness

    with LLMMockHarness.happy_path() as h:
        yield h
