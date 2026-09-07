from __future__ import annotations

import json

from unittest.mock import AsyncMock, patch

import pytest

from finagents.dataflows import scrip_master
from finagents.dataflows.scrip_master import InstrumentNotFound




@pytest.mark.asyncio
async def test_cold_cache_fetches_and_stores(
    httpx_mock, recorded_fixture_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    fixture = (recorded_fixture_dir / "angelone" / "scrip_master_sample.json").read_bytes()
    httpx_mock.add_response(url=scrip_master.SCRIP_MASTER_URL, content=fixture)

    result = await scrip_master.get_scrip_master()

    assert isinstance(result, list)
    assert any(e["symbol"] == "RELIANCE-EQ" for e in result)
    assert any(e["symbol"] == "INFY-EQ" for e in result)
    assert any(e["symbol"] == "TCS-EQ" for e in result)




@pytest.mark.asyncio
async def test_warm_cache_no_network_call(recorded_fixture_dir, tmp_path, monkeypatch):
    from finagents.dataflows import cache as cache_mod

    db_path = tmp_path / "t.db"
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", db_path)
    fixture = (recorded_fixture_dir / "angelone" / "scrip_master_sample.json").read_bytes()

    await cache_mod.set_cached(
        scrip_master.CATEGORY, scrip_master.CACHE_KEY, fixture, db_path=db_path
    )

    result = await scrip_master.get_scrip_master()
    assert isinstance(result, list)
    assert any(e["symbol"] == "RELIANCE-EQ" for e in result)




@pytest.mark.asyncio
async def test_stale_cache_refetches(httpx_mock, recorded_fixture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    fixture = (recorded_fixture_dir / "angelone" / "scrip_master_sample.json").read_bytes()

    with patch("finagents.dataflows.scrip_master.cache") as mock_cache:
        mock_cache.get_cached = AsyncMock(return_value=(fixture, True))
        mock_cache.set_cached = AsyncMock()

        httpx_mock.add_response(url=scrip_master.SCRIP_MASTER_URL, content=fixture)
        result = await scrip_master.get_scrip_master()

    assert isinstance(result, list)
    assert mock_cache.set_cached.call_count == 1




@pytest.mark.asyncio
async def test_resolve_token_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    fixture_data = [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5",
        },
    ]

    with patch("finagents.dataflows.scrip_master._master_bytes") as mock_get:
        mock_get.return_value = json.dumps(fixture_data).encode()

        token_upper = await scrip_master.resolve_token("RELIANCE")
        assert token_upper == "2885"

        token_lower = await scrip_master.resolve_token("reliance")
        assert token_lower == "2885"

        token_with_eq = await scrip_master.resolve_token("RELIANCE-EQ")
        assert token_with_eq == "2885"




@pytest.mark.asyncio
async def test_instrument_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    fixture_data = [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5",
        },
    ]

    with patch("finagents.dataflows.scrip_master._master_bytes") as mock_get:
        mock_get.return_value = json.dumps(fixture_data).encode()

        with pytest.raises(InstrumentNotFound, match="NOTAREALTICKER"):
            await scrip_master.resolve_token("NOTAREALTICKER")




def test_scrip_master_url_domain():
    assert (
        "margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
        in scrip_master.SCRIP_MASTER_URL
    )
    assert "angelbroking.com" not in scrip_master.SCRIP_MASTER_URL




@pytest.mark.asyncio
async def test_exchange_filter(tmp_path, monkeypatch):
    fixture_data = [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5",
        },
        {
            "token": "9999",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "BSE",
            "tick_size": "5",
        },
    ]

    with patch("finagents.dataflows.scrip_master._master_bytes") as mock_get:
        mock_get.return_value = json.dumps(fixture_data).encode()

        token_nse = await scrip_master.resolve_token("RELIANCE", exchange="NSE")
        assert token_nse == "2885"

        token_bse = await scrip_master.resolve_token("RELIANCE", exchange="BSE")
        assert token_bse == "9999"


async def test_token_index_reused_and_invalidated(monkeypatch):
    original = json.dumps([{"symbol": "A-EQ", "token": "1", "exch_seg": "NSE"}]).encode()
    changed = original.replace(b'"1"', b'"2"')
    loader = AsyncMock(side_effect=[original, original, changed])
    monkeypatch.setattr(scrip_master, "_master_bytes", loader)
    scrip_master._token_index.cache_clear()
    assert await scrip_master.resolve_token("A") == "1"
    assert await scrip_master.resolve_token("a") == "1"
    assert scrip_master._token_index.cache_info().hits == 1
    assert await scrip_master.resolve_token("A") == "2"
    assert scrip_master._token_index.cache_info().misses == 2
