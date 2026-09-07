from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from finagents.dataflows.fundamentals_schema import FundamentalsReport



FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _read_fixture(rel_path: str) -> bytes:
    return (FIXTURE_DIR / rel_path).read_bytes()




@pytest.mark.asyncio
async def test_csv_path_success(httpx_mock, monkeypatch, tmp_path):
    from finagents.dataflows import cache, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    csv_bytes = _read_fixture("screener/reliance.csv")
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=csv_bytes,
        headers={"content-type": "text/csv; charset=utf-8"},
    )

    from finagents.dataflows import fundamentals_client

    result = await fundamentals_client.get_fundamentals("RELIANCE")

    assert result.status in ("ok", "partial")
    assert result.source == "screener_csv"
    assert result.data is not None
    assert result.data.valuation.pe_ratio == 29.9
    assert result.data.profitability.roce == 13.5
    assert result.data.shareholding.promoter_holding_pct == 50.3




@pytest.mark.asyncio
async def test_csv_returns_login_html_falls_through(httpx_mock, monkeypatch, tmp_path):
    from finagents.dataflows import cache, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    login_html = _read_fixture("screener/login_redirect.html")
    reliance_html = _read_fixture("screener/reliance.html")

    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=login_html,
        headers={"content-type": "text/html; charset=utf-8"},
    )
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/",
        content=reliance_html,
        headers={"content-type": "text/html; charset=utf-8"},
    )

    from finagents.dataflows import fundamentals_client

    result = await fundamentals_client.get_fundamentals("RELIANCE")

    assert result.source == "screener_html"
    assert result.data is not None




@pytest.mark.asyncio
async def test_ticker_sanitization(monkeypatch, tmp_path):
    from finagents.dataflows import cache

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")

    from finagents.dataflows import fundamentals_client

    for bad_ticker in ["REL/../etc/passwd", "RELIANCE?foo=bar", "reliance; drop", ""]:
        result = await fundamentals_client.get_fundamentals(bad_ticker)
        assert result.status == "error", f"Expected error for {bad_ticker!r}"
        assert result.error is not None
        assert "Invalid ticker" in result.error or "invalid" in (result.error or "").lower(), (
            f"Expected 'Invalid ticker' in error for {bad_ticker!r}, got {result.error!r}"
        )

    safe = fundamentals_client._sanitize_ticker("RELIANCE")
    assert safe == "RELIANCE"




@pytest.mark.asyncio
async def test_rate_limit_acquired(httpx_mock, monkeypatch, tmp_path):
    from finagents.dataflows import cache, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")

    mock_acquire = AsyncMock()
    monkeypatch.setattr(rate_limits, "acquire", mock_acquire)

    csv_bytes = _read_fixture("screener/reliance.csv")
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=csv_bytes,
        headers={"content-type": "text/csv"},
    )

    from finagents.dataflows import fundamentals_client

    await fundamentals_client.get_fundamentals("RELIANCE")

    calls = [str(c) for c in mock_acquire.call_args_list]
    assert any("screener.in" in c for c in calls), (
        f"Expected acquire('screener.in') call, got: {mock_acquire.call_args_list}"
    )




def test_html_selectors(tmp_path):
    from finagents.dataflows.fundamentals_client import _parse_screener_html

    html_bytes = _read_fixture("screener/reliance.html")
    report = _parse_screener_html(html_bytes, "RELIANCE")

    assert isinstance(report, FundamentalsReport)
    assert report.valuation.pe_ratio == 29.9
    assert report.valuation.pb_ratio == 2.2
    assert report.profitability.roce == 13.5
    assert report.profitability.roe == 9.8
    assert report.debt_balance.debt_to_equity == 0.45
    assert report.debt_balance.current_ratio == 1.35
    assert report.shareholding.promoter_holding_pct == 50.3
    assert report.shareholding.fii_holding_pct == 22.1




@pytest.mark.asyncio
async def test_cache_hit_skips_network(httpx_mock, monkeypatch, tmp_path):
    from finagents.dataflows import cache, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    csv_bytes = _read_fixture("screener/reliance.csv")
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=csv_bytes,
        headers={"content-type": "text/csv"},
    )

    from finagents.dataflows import fundamentals_client

    result1 = await fundamentals_client.get_fundamentals("RELIANCE")
    assert result1.source == "screener_csv"

    result2 = await fundamentals_client.get_fundamentals("RELIANCE")
    assert result2.source == "cache"
    assert result2.data is not None
    assert result2.data.valuation.pe_ratio == 29.9




@pytest.mark.asyncio
async def test_partial_status_when_shareholding_missing(httpx_mock, monkeypatch, tmp_path):
    from finagents.dataflows import cache, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    minimal_csv = b"""Company Name,RELIANCE INDUSTRIES
Price to Earning,29.9
Price to book value,2.2
Return on capital employed,13.5
Debt to equity,0.45
"""
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=minimal_csv,
        headers={"content-type": "text/csv"},
    )

    from finagents.dataflows import fundamentals_client

    result = await fundamentals_client.get_fundamentals("RELIANCE")

    assert result.status == "partial"
    assert result.data is not None
    assert result.data.shareholding.promoter_holding_pct is None




def test_ticker_whitelist_pattern():
    from finagents.dataflows.fundamentals_client import _sanitize_ticker

    assert _sanitize_ticker("RELIANCE") == "RELIANCE"
    assert _sanitize_ticker("TATA-MOTORS") == "TATA-MOTORS"
    assert _sanitize_ticker("M&M") == "M&M"
    assert _sanitize_ticker("HDFC") == "HDFC"
    assert _sanitize_ticker("reliance") == "RELIANCE"

    import pytest as pt

    for bad in ["RELIANCE;DROP", "foo/bar", "RELIANCE ", "", "A" * 21, "rel iance"]:
        with pt.raises(ValueError, match="Invalid ticker"):
            _sanitize_ticker(bad)




@pytest.mark.asyncio
async def test_accepts_settings_parameter(httpx_mock, monkeypatch, tmp_path):
    from finagents.config import Settings
    from finagents.dataflows import cache, fundamentals_client, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    csv_bytes = _read_fixture("screener/reliance.csv")
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=csv_bytes,
        headers={"content-type": "text/csv; charset=utf-8"},
    )

    result = await fundamentals_client.get_fundamentals(
        "RELIANCE", Settings(openrouter_api_key="fake")
    )
    assert result.status in ("ok", "partial")
    assert result.source == "screener_csv"
    assert result.data is not None

    result_kw = await fundamentals_client.get_fundamentals(
        "RELIANCE", settings=Settings(openrouter_api_key="fake")
    )
    assert result_kw.source == "cache"
    assert result_kw.data is not None

    import inspect

    sig = inspect.signature(fundamentals_client.get_fundamentals)
    assert "settings" in sig.parameters
    assert sig.parameters["settings"].default is None
