from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from finagents.dataflows import cache, surveillance_client

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "nse"
SAMPLE_FIXTURE = FIXTURE_DIR / "sec_list_sample.csv"


@pytest.mark.asyncio
async def test_cold_cache_fetches_and_stores(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    fixture = SAMPLE_FIXTURE.read_bytes()
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=fixture,
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("TRENT")
    assert circuit_pct == pytest.approx(0.20)
    assert t2t is False


@pytest.mark.asyncio
async def test_no_band_returns_none_for_circuit_pct(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("RELIANCE")
    assert circuit_pct is None
    assert t2t is False


@pytest.mark.asyncio
async def test_band_20_parses_to_0_20(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("ZOMATO")
    assert circuit_pct == pytest.approx(0.20)
    assert t2t is False


@pytest.mark.asyncio
async def test_band_5_parses_to_0_05(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("AAREYDRUGS")
    assert circuit_pct == pytest.approx(0.05)
    assert t2t is False


@pytest.mark.asyncio
async def test_series_be_is_t2t(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("BLUECHIP")
    assert circuit_pct == pytest.approx(0.02)
    assert t2t is True


@pytest.mark.asyncio
async def test_series_bz_is_t2t(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("ANSALAPI")
    assert circuit_pct == pytest.approx(0.02)
    assert t2t is True


@pytest.mark.asyncio
async def test_series_st_is_t2t(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("FAKESURV")
    assert circuit_pct == pytest.approx(0.10)
    assert t2t is True


@pytest.mark.asyncio
async def test_ticker_not_in_list_returns_safe_default(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("DOESNOTEXIST")
    assert circuit_pct is None
    assert t2t is False


@pytest.mark.asyncio
async def test_fetch_failure_returns_safe_default(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        status_code=500,
        content=b"Internal Server Error",
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("TRENT")
    assert circuit_pct is None
    assert t2t is False


@pytest.mark.asyncio
async def test_warm_cache_skips_http(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", db_path)

    fixture = SAMPLE_FIXTURE.read_bytes()
    await cache.set_cached(
        surveillance_client.CATEGORY,
        surveillance_client.CACHE_KEY,
        fixture,
        db_path=db_path,
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("TRENT")
    assert circuit_pct == pytest.approx(0.20)
    assert t2t is False

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("ANSALAPI")
    assert circuit_pct == pytest.approx(0.02)
    assert t2t is True


@pytest.mark.asyncio
async def test_ticker_lookup_is_case_insensitive(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_response(
        url=surveillance_client.NSE_SEC_LIST_URL,
        content=SAMPLE_FIXTURE.read_bytes(),
    )

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("trent")
    assert circuit_pct == pytest.approx(0.20)
    assert t2t is False


@pytest.mark.asyncio
async def test_network_exception_returns_safe_default(httpx_mock, tmp_path, monkeypatch):
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "t.db")
    httpx_mock.add_exception(httpx.ConnectTimeout("timed out"))

    circuit_pct, t2t = await surveillance_client.get_surveillance_data("TRENT")
    assert circuit_pct is None
    assert t2t is False
