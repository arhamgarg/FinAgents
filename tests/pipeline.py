from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from finagents.dataflows.data_result import DataResult
from finagents.dataflows.fundamentals_client import _parse_screener_csv
from finagents.dataflows.news_schema import NewsArticle
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
    fundamentals = _parse_screener_csv(
        (Path(__file__).parent / "fixtures/screener/reliance.csv").read_bytes(), "RELIANCE"
    )
    articles = [
        NewsArticle(
            title=f"Recorded news {i}", url=f"https://example.invalid/{i}", source="fixture"
        )
        for i in range(6)
    ]
    with (
        patch(
            "finagents.agents.analysts.technical_analyst.get_ohlcv",
            new=AsyncMock(return_value=DataResult(status="ok", data=ohlcv_250(), source="fixture")),
        ),
        patch(
            "finagents.agents.analysts.technical_analyst.get_surveillance_data",
            new=AsyncMock(return_value=(0.20, False)),
        ),
        patch(
            "finagents.agents.analysts.fundamentals_analyst.get_fundamentals",
            new=AsyncMock(
                return_value=DataResult(status="ok", data=fundamentals, source="fixture")
            ),
        ),
        patch(
            "finagents.agents.analysts.news_analyst.get_news",
            new=AsyncMock(return_value=DataResult(status="ok", data=articles, source="fixture")),
        ),
        LLMMockHarness.happy_path() as harness,
    ):
        yield harness
