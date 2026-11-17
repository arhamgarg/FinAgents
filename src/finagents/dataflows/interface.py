from __future__ import annotations

from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from finagents.dataflows.fundamentals_client import get_fundamentals
from finagents.dataflows.news_client import get_news
from finagents.dataflows.ohlcv_client import get_ohlcv
from finagents.dataflows.surveillance_client import get_surveillance_data

if TYPE_CHECKING:
    import pandas as pd

    from finagents.config import Settings
    from finagents.dataflows.data_result import DataResult
    from finagents.dataflows.fundamentals_schema import FundamentalsReport
    from finagents.dataflows.news_schema import NewsArticle
    from finagents.dataflows.ohlcv_client import Interval


class OhlcvSource(Protocol):
    def __call__(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = "ONE_DAY",
        *,
        settings: Settings,
    ) -> Awaitable[DataResult[pd.DataFrame]]: ...


class FundamentalsSource(Protocol):
    def __call__(
        self, ticker: str, settings: Settings
    ) -> Awaitable[DataResult[FundamentalsReport]]: ...


class NewsSource(Protocol):
    def __call__(
        self, ticker: str, settings: Settings
    ) -> Awaitable[DataResult[list[NewsArticle]]]: ...


class SurveillanceSource(Protocol):
    def __call__(self, ticker: str) -> Awaitable[tuple[float | None, bool]]: ...


@dataclass(frozen=True, kw_only=True)
class DataSources:
    ohlcv: OhlcvSource = get_ohlcv
    fundamentals: FundamentalsSource = get_fundamentals
    news: NewsSource = get_news
    surveillance: SurveillanceSource = get_surveillance_data


DEFAULT_DATA_SOURCES = DataSources()

_current = DEFAULT_DATA_SOURCES


def data_sources() -> DataSources:
    return _current


@contextmanager
def use_data_sources(
    *,
    ohlcv: OhlcvSource | None = None,
    fundamentals: FundamentalsSource | None = None,
    news: NewsSource | None = None,
    surveillance: SurveillanceSource | None = None,
) -> Iterator[DataSources]:
    global _current  # noqa: PLW0603
    previous = _current
    _current = replace(
        previous,
        ohlcv=previous.ohlcv if ohlcv is None else ohlcv,
        fundamentals=previous.fundamentals if fundamentals is None else fundamentals,
        news=previous.news if news is None else news,
        surveillance=previous.surveillance if surveillance is None else surveillance,
    )
    try:
        yield _current
    finally:
        _current = previous


__all__ = [
    "DEFAULT_DATA_SOURCES",
    "DataSources",
    "FundamentalsSource",
    "NewsSource",
    "OhlcvSource",
    "SurveillanceSource",
    "data_sources",
    "get_fundamentals",
    "get_news",
    "get_ohlcv",
    "use_data_sources",
]
