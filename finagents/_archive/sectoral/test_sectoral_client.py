from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from finagents.dataflows.sectoral_client import (
    REQUIRED_SECTORAL,
    SectoralIndexSnapshot,
    get_sectoral_indices,
    is_market_hours,
    sectoral_ttl_seconds,
)
from finagents.dataflows.sectoral_client import (
    NSE_ALL_INDICES,
    NSE_HOME,
)

IST = ZoneInfo("Asia/Kolkata")

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "nse"




@pytest.mark.asyncio
async def test_cookie_handshake_sequence(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()
    all_indices_payload = json.loads((FIXTURE_DIR / "allIndices.json").read_text())

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json=all_indices_payload, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        await get_sectoral_indices()

    requests = httpx_mock.get_requests()
    assert len(requests) == 2, f"Expected 2 requests, got {len(requests)}"
    assert "/api/" not in str(requests[0].url), (
        f"First request should be homepage, not API: {requests[0].url}"
    )
    assert "/api/allIndices" in str(requests[1].url), (
        f"Second request should be allIndices API: {requests[1].url}"
    )




@pytest.mark.asyncio
async def test_nine_indices_returned(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()
    all_indices_payload = json.loads((FIXTURE_DIR / "allIndices.json").read_text())

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json=all_indices_payload, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        result = await get_sectoral_indices()

    assert result.status == "ok", f"Expected ok, got {result.status}: {result.error}"
    assert result.data is not None
    assert set(result.data.keys()) == REQUIRED_SECTORAL, (
        f"Missing: {REQUIRED_SECTORAL - set(result.data.keys())}, "
        f"Extra: {set(result.data.keys()) - REQUIRED_SECTORAL}"
    )
    assert "NIFTY 50" not in result.data
    assert "NIFTY MIDCAP 100" not in result.data




def test_market_hours_ttl():
    tuesday_market = datetime(2024, 1, 2, 10, 30, tzinfo=IST)
    tuesday_evening = datetime(2024, 1, 2, 18, 0, tzinfo=IST)
    saturday = datetime(2024, 1, 6, 10, 30, tzinfo=IST)

    assert sectoral_ttl_seconds(tuesday_market) == 300, "Market hours should be 5 min TTL"
    assert sectoral_ttl_seconds(tuesday_evening) == 86400, "After hours should be 24h TTL"
    assert sectoral_ttl_seconds(saturday) == 86400, "Weekend should be 24h TTL"

    at_open = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    at_close = datetime(2024, 1, 2, 15, 30, tzinfo=IST)
    before_open = datetime(2024, 1, 2, 9, 14, tzinfo=IST)
    after_close = datetime(2024, 1, 2, 15, 31, tzinfo=IST)

    assert is_market_hours(at_open) is True
    assert is_market_hours(at_close) is True
    assert is_market_hours(before_open) is False
    assert is_market_hours(after_close) is False




@pytest.mark.asyncio
async def test_snapshot_fields(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()
    all_indices_payload = json.loads((FIXTURE_DIR / "allIndices.json").read_text())

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json=all_indices_payload, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        result = await get_sectoral_indices()

    assert result.status == "ok"
    bank = result.data["NIFTY BANK"]
    assert isinstance(bank, SectoralIndexSnapshot)
    assert bank.last == 47523.45
    assert bank.percent_change == 0.26
    assert bank.previous_close == 47399.95
    assert bank.year_high == 49794.70
    assert bank.year_low == 41785.75




@pytest.mark.asyncio
async def test_empty_response_is_empty_status(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json={"data": []}, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        result = await get_sectoral_indices()

    assert result.status == "empty"
    assert result.source == "nse"
    assert result.data is None




@pytest.mark.asyncio
async def test_rate_limiter_acquired(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()
    all_indices_payload = json.loads((FIXTURE_DIR / "allIndices.json").read_text())

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json=all_indices_payload, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        await get_sectoral_indices()

    calls = mock_rl.acquire.call_args_list
    assert len(calls) >= 2, f"Expected at least 2 acquire calls, got {len(calls)}"
    for call in calls:
        assert call.args[0] == "www.nseindia.com", (
            f"acquire called with unexpected host: {call.args[0]}"
        )




@pytest.mark.asyncio
async def test_cache_write_on_success(httpx_mock, tmp_path, monkeypatch):
    from finagents.dataflows import cache as cache_mod

    db_path = tmp_path / "t.db"
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", db_path)

    homepage_html = (FIXTURE_DIR / "homepage.html").read_bytes()
    all_indices_payload = json.loads((FIXTURE_DIR / "allIndices.json").read_text())

    httpx_mock.add_response(url=NSE_HOME, content=homepage_html, status_code=200)
    httpx_mock.add_response(url=NSE_ALL_INDICES, json=all_indices_payload, status_code=200)

    with patch("finagents.dataflows.sectoral_client.rate_limits") as mock_rl:
        mock_rl.acquire = AsyncMock()
        result = await get_sectoral_indices()

    assert result.status == "ok"
    cached = await cache_mod.get_cached("sectoral", "allIndices", db_path=db_path)
    assert cached is not None, "Cache should have been written after successful fetch"
    blob, stale = cached
    assert blob is not None
