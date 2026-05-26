from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from financialagents.dataflows.data_result import DataResult

INTERFACE_FUNCTIONS = [
    "get_ohlcv",
    "get_fundamentals",
    "get_news",
]




def test_interface_exposes_all_functions():
    from financialagents.dataflows import interface

    for name in INTERFACE_FUNCTIONS:
        assert hasattr(interface, name), f"interface.{name} is missing"




@pytest.mark.asyncio
async def test_get_ohlcv_delegates():
    from financialagents.dataflows import interface

    fake_df = pd.DataFrame(
        {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    )
    fake_result = DataResult[pd.DataFrame](status="ok", data=fake_df, source="angelone")
    mock_fn = AsyncMock(return_value=fake_result)
    monkeypatch_target = "financialagents.dataflows.interface.ohlcv_client.get_ohlcv"

    with patch(monkeypatch_target, mock_fn):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        result = await interface.get_ohlcv("RELIANCE", start, end, "ONE_DAY")

    assert result is fake_result
    mock_fn.assert_called_once_with("RELIANCE", start, end, "ONE_DAY", None)




@pytest.mark.asyncio
async def test_get_fundamentals_delegates():
    from financialagents.dataflows import interface
    from financialagents.dataflows.fundamentals_schema import (
        DebtBalance,
        FundamentalsReport,
        Profitability,
        Shareholding,
        Valuation,
    )

    fake_report = FundamentalsReport(
        ticker="RELIANCE",
        valuation=Valuation(),
        profitability=Profitability(),
        debt_balance=DebtBalance(),
        shareholding=Shareholding(),
    )
    fake_result = DataResult[FundamentalsReport](
        status="ok", data=fake_report, source="screener_csv"
    )
    mock_fn = AsyncMock(return_value=fake_result)

    with patch("financialagents.dataflows.interface.fundamentals_client.get_fundamentals", mock_fn):
        result = await interface.get_fundamentals("RELIANCE")

    assert result is fake_result
    mock_fn.assert_called_once_with("RELIANCE", None)




@pytest.mark.asyncio
async def test_get_news_delegates():
    from financialagents.dataflows import interface
    from financialagents.dataflows.news_schema import NewsArticle

    fake_result = DataResult[list[NewsArticle]](status="ok", data=[], source="moneycontrol")
    mock_fn = AsyncMock(return_value=fake_result)

    with patch("financialagents.dataflows.interface.news_client.get_news", mock_fn):
        result = await interface.get_news("RELIANCE")

    assert result is fake_result
    mock_fn.assert_called_once_with("RELIANCE")




def test_all_functions_are_async():
    from financialagents.dataflows import interface

    for name in INTERFACE_FUNCTIONS:
        fn = getattr(interface, name)
        assert inspect.iscoroutinefunction(fn), (
            f"interface.{name} is not async — violates async-first requirement"
        )




def test_module_docstring_mentions_d06():
    from pathlib import Path

    src = Path("financialagents/dataflows/interface.py").read_text()
    assert "" in src or "data_vendors" in src, (
        "interface.py must document / data_vendors routing — "
        " authors need to understand why routing isn't done here"
    )




@pytest.mark.asyncio
async def test_get_fundamentals_delegates_integration(
    httpx_mock, monkeypatch, tmp_path
):
    from pathlib import Path

    from financialagents.config import Settings
    from financialagents.dataflows import cache, interface, rate_limits

    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rate_limits, "acquire", AsyncMock())

    fixture_path = (
        Path(__file__).parent.parent / "fixtures" / "screener" / "reliance.csv"
    )
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
