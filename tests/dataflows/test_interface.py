from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_get_fundamentals_delegates_integration(httpx_mock, monkeypatch, tmp_path):
    from pathlib import Path

    from finagents.config import Settings
    from finagents.dataflows import cache, interface, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    fixture_path = Path(__file__).parent.parent / "fixtures" / "screener" / "reliance.csv"
    csv_bytes = fixture_path.read_bytes()
    httpx_mock.add_response(
        url="https://www.screener.in/company/RELIANCE/consolidated/export/",
        content=csv_bytes,
        headers={"content-type": "text/csv; charset=utf-8"},
    )

    result = await interface.get_fundamentals("RELIANCE", Settings(openrouter_api_key="fake"))

    assert result.status in ("ok", "partial")
    assert result.source == "screener_csv"
    assert result.data is not None
    assert result.data.ticker == "RELIANCE"
