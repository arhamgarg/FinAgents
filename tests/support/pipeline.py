from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd

from finagents.dataflows.data_result import DataResult
from finagents.dataflows.interface import use_data_sources
from finagents.dataflows.news_schema import NewsArticle
from finagents.dataflows.screener_csv import parse_screener_csv
from finagents.testing.llm_mock import LLMMockHarness


def ohlcv_250():
    rng = np.random.default_rng(42)
    price = 2500 + np.cumsum(rng.normal(0, 2, 250))
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price,
            "volume": rng.integers(100_000, 1_000_000, 250),
        },
        index=pd.date_range("2024-01-01", periods=250, freq="B"),
    )


@contextmanager
def pipeline_scenario():
    fundamentals = parse_screener_csv(
        (Path(__file__).parents[1] / "fixtures/screener/reliance.csv").read_bytes(), "RELIANCE"
    )
    articles = [
        NewsArticle(
            title=f"Recorded news {i}", url=f"https://example.invalid/{i}", source="fixture"
        )
        for i in range(6)
    ]
    with (
        use_data_sources(
            ohlcv=AsyncMock(
                return_value=DataResult(status="ok", data=ohlcv_250(), source="fixture")
            ),
            surveillance=AsyncMock(return_value=(0.20, False)),
            fundamentals=AsyncMock(
                return_value=DataResult(status="ok", data=fundamentals, source="fixture")
            ),
            news=AsyncMock(return_value=DataResult(status="ok", data=articles, source="fixture")),
        ),
        LLMMockHarness.happy_path() as harness,
    ):
        yield harness
