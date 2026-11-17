from __future__ import annotations

import asyncio
import random

import pytest

import finagents.dataflows.rate_limits as rate_limits_module
from finagents.config import Settings


class TestRateLimits:
    def test_default_screener_min_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCREENER_MIN_DELAY", raising=False)
        assert rate_limits_module.min_delays(Settings())["screener.in"] == 4.0

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCREENER_MIN_DELAY", "7.5")
        assert rate_limits_module.min_delays(Settings())["screener.in"] == 7.5

    def test_all_expected_hosts_present(self) -> None:
        expected_hosts = {
            "screener.in",
            "www.moneycontrol.com",
            "economictimes.indiatimes.com",
            "www.livemint.com",
            "www.nseindia.com",
        }
        assert set(rate_limits_module.min_delays(Settings())) == expected_hosts

    @pytest.mark.asyncio
    async def test_unknown_host_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            await rate_limits_module.acquire("unknown.example.com", Settings())

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

        await rate_limits_module.acquire("www.nseindia.com", Settings())

        assert len(uniform_calls) >= 1
        last_call = uniform_calls[-1]
        assert last_call == (0.0, Settings().rate_jitter_max)

    @pytest.mark.asyncio
    async def test_acquire_back_to_back_uses_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleep_calls = []

        async def mock_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        await rate_limits_module.acquire("www.nseindia.com", Settings())
        assert len(sleep_calls) == 1
