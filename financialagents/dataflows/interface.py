from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from financialagents.config import Settings
from financialagents.dataflows import (
    fundamentals_client,
    news_client,
    ohlcv_client,
)
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.fundamentals_schema import FundamentalsReport
from financialagents.dataflows.news_schema import NewsArticle
from financialagents.dataflows.ohlcv_client import Interval


async def get_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: Interval = "ONE_DAY",
    settings: Optional[Settings] = None,
) -> DataResult[pd.DataFrame]:
    return await ohlcv_client.get_ohlcv(ticker, start, end, interval, settings)


async def get_fundamentals(
    ticker: str,
    settings: Optional[Settings] = None,
) -> DataResult[FundamentalsReport]:
    return await fundamentals_client.get_fundamentals(ticker, settings)


async def get_news(ticker: str) -> DataResult[list[NewsArticle]]:
    return await news_client.get_news(ticker)


__all__ = [
    "get_ohlcv",
    "get_fundamentals",
    "get_news",
]
