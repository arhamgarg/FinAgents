from __future__ import annotations

import asyncio
import importlib
import random
from unittest.mock import MagicMock, patch, call

import pytest

import financialagents.dataflows.rate_limits as rate_limits_module


class TestRateLimits:
    def test_default_screener_min_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCREENER_MIN_DELAY", raising=False)
        importlib.reload(rate_limits_module)
        assert rate_limits_module.SCREENER_MIN_DELAY == 4.0

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCREENER_MIN_DELAY", "7.5")
        importlib.reload(rate_limits_module)
        assert rate_limits_module.SCREENER_MIN_DELAY == 7.5
        monkeypatch.delenv("SCREENER_MIN_DELAY", raising=False)
        importlib.reload(rate_limits_module)

    def test_all_seven_hosts_present(self) -> None:
        expected_hosts = {
            "screener.in",
            "www.moneycontrol.com",
            "economictimes.indiatimes.com",
            "www.livemint.com",
            "www.nseindia.com",
            "api.stocktwits.com",
            "trends.google.com",
        }
        assert set(rate_limits_module._LIMITERS.keys()) == expected_hosts

    @pytest.mark.asyncio
    async def test_unknown_host_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            await rate_limits_module.acquire("unknown.example.com")

    @pytest.mark.asyncio
    async def test_jitter_uses_random_uniform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleep_calls = []

        async def mock_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        uniform_calls = []

        def mock_uniform(a: float, b: float) -> float:
            uniform_calls.append((a, b))
            return 0.0

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "uniform", mock_uniform)

        await rate_limits_module.acquire("www.nseindia.com")

        assert len(uniform_calls) >= 1
        last_call = uniform_calls[-1]
        assert last_call == (0.0, rate_limits_module.JITTER_MAX)

    @pytest.mark.asyncio
    async def test_acquire_back_to_back_uses_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleep_calls = []

        async def mock_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        await rate_limits_module.acquire("www.nseindia.com")
        assert len(sleep_calls) == 1
