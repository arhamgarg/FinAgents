from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.config import Settings
from financialagents.dataflows import sentiment_client
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.sentiment_schema import (
    RedditSentiment,
    StockTwitsSentiment,
    TrendsSentiment,
)

FIXTURES_DIR = Path("tests/fixtures/sentiment")
STOCKTWITS_URL_EMPTY = "https://api.stocktwits.com/api/2/streams/symbol/RELIANCE.json"
STOCKTWITS_URL_INFY = "https://api.stocktwits.com/api/2/streams/symbol/INFY.json"




@pytest.mark.asyncio
async def test_missing_reddit_creds_empty():
    settings = Settings.model_construct(
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent="test/1.0",
        data_vendors={},
    )
    with (
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock) as mock_st,
        patch.object(sentiment_client, "_fetch_trends", new_callable=AsyncMock) as mock_tr,
    ):
        mock_st.return_value = StockTwitsSentiment()
        mock_tr.return_value = TrendsSentiment(ticker="RELIANCE")

        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert isinstance(result, DataResult)
    assert result.data is not None
    assert result.data.sources_status.get("reddit") == "empty"
    assert result.status in ("empty", "ok", "partial")
    assert result.status != "error"




@pytest.mark.asyncio
async def test_praw_wrapped_in_to_thread():
    settings = Settings.model_construct(
        reddit_client_id="fake_id",
        reddit_client_secret="fake_secret",
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    captured_calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        captured_calls.append(fn)
        return []

    with patch("financialagents.dataflows.sentiment_client.asyncio") as mock_asyncio:
        mock_asyncio.to_thread = AsyncMock(side_effect=fake_to_thread)
        result = await sentiment_client._fetch_reddit(settings, "RELIANCE")

    assert len(captured_calls) >= 1, "asyncio.to_thread was not called — PRAW blocking the event loop!"
    assert isinstance(result, RedditSentiment)




@pytest.mark.asyncio
async def test_stocktwits_empty_is_ok():
    payload = json.loads((FIXTURES_DIR / "stocktwits_empty_reliance.json").read_text())

    settings = Settings.model_construct(
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    messages_raw = payload.get("messages", []) or []
    expected_result = StockTwitsSentiment(
        messages=[],
        bullish_count=0,
        bearish_count=0,
        neutral_count=0,
    )

    with (
        patch.object(sentiment_client, "_fetch_reddit", side_effect=ValueError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")),
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock, return_value=expected_result),
        patch.object(sentiment_client, "_fetch_trends", new_callable=AsyncMock, return_value=TrendsSentiment(ticker="RELIANCE")),
    ):
        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert result.data is not None
    assert result.data.sources_status.get("stocktwits") == "ok"
    assert result.data.stocktwits.messages == []




@pytest.mark.asyncio
async def test_stocktwits_populated():
    import json as _json

    payload = _json.loads((FIXTURES_DIR / "stocktwits_stream.json").read_text())

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = payload

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch.object(sentiment_client.rate_limits, "acquire", new_callable=AsyncMock):
        result = await sentiment_client._fetch_stocktwits(mock_client, "INFY")

    assert result.bullish_count == 1
    assert result.bearish_count == 1
    assert result.neutral_count == 1
    assert len(result.messages) == 3




def test_trendspyg_not_pytrends():
    src = Path("financialagents/dataflows/sentiment_client.py").read_text()
    assert "from trendspyg" in src, "Expected 'from trendspyg' import in sentiment_client.py"
    assert "import pytrends" not in src, "Forbidden: pytrends is archived (research drift #1)"
    assert "from pytrends" not in src, "Forbidden: pytrends is archived (research drift #1)"




@pytest.mark.asyncio
async def test_trendspyg_failure_degrades():
    settings = Settings.model_construct(
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    async def failing_trends(ticker: str) -> TrendsSentiment:
        raise RuntimeError("trendspyg connection error")

    with (
        patch.object(sentiment_client, "_fetch_reddit", new_callable=AsyncMock) as mock_r,
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock) as mock_st,
        patch.object(sentiment_client, "_fetch_trends", side_effect=RuntimeError("trendspyg fail")),
    ):
        mock_r.side_effect = ValueError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        mock_st.return_value = StockTwitsSentiment()

        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert result.data is not None
    assert result.data.sources_status.get("trends") == "error"
    assert result.status != "error"




@pytest.mark.asyncio
async def test_all_three_sources_merged():
    from datetime import datetime, timezone

    from financialagents.dataflows.sentiment_schema import RedditPost, StockTwitsMessage

    settings = Settings.model_construct(
        reddit_client_id="fake",
        reddit_client_secret="fake",
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    fake_reddit = RedditSentiment(
        posts=[
            RedditPost(
                title="Test post",
                score=100,
                url="https://reddit.com/r/test",
                created_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
                subreddit="IndianStreetBets",
            )
        ],
        post_count=1,
        avg_score=100.0,
    )
    fake_stocktwits = StockTwitsSentiment(
        messages=[
            StockTwitsMessage(
                id=1,
                body="Bullish on RELIANCE",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                sentiment="bullish",
            )
        ],
        bullish_count=1,
        bearish_count=0,
        neutral_count=0,
    )
    fake_trends = TrendsSentiment(
        ticker="RELIANCE",
        interest_over_time=[(datetime(2024, 1, 1, tzinfo=timezone.utc), 65)],
        peak_interest=65,
        avg_interest=65.0,
    )

    with (
        patch.object(sentiment_client, "_fetch_reddit", new_callable=AsyncMock, return_value=fake_reddit),
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock, return_value=fake_stocktwits),
        patch.object(sentiment_client, "_fetch_trends", new_callable=AsyncMock, return_value=fake_trends),
    ):
        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.reddit.post_count == 1
    assert result.data.stocktwits.bullish_count == 1
    assert result.data.trends is not None
    assert len(result.data.trends.interest_over_time) == 1
    assert result.data.sources_status.get("reddit") == "ok"
    assert result.data.sources_status.get("stocktwits") == "ok"
    assert result.data.sources_status.get("trends") == "ok"




@pytest.mark.asyncio
async def test_overall_status_empty_when_all_empty():
    settings = Settings.model_construct(
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    with (
        patch.object(sentiment_client, "_fetch_reddit", side_effect=ValueError("no creds")),
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock, return_value=StockTwitsSentiment()),
        patch.object(sentiment_client, "_fetch_trends", new_callable=AsyncMock, return_value=TrendsSentiment(ticker="RELIANCE")),
    ):
        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert result.status == "empty"
    assert result.data is not None




@pytest.mark.asyncio
async def test_overall_never_errors_when_any_source_has_data():
    from datetime import datetime, timezone

    from financialagents.dataflows.sentiment_schema import StockTwitsMessage

    settings = Settings.model_construct(
        reddit_client_id="fake",
        reddit_client_secret="fake",
        reddit_user_agent="test/1.0",
        data_vendors={},
    )

    fake_stocktwits = StockTwitsSentiment(
        messages=[
            StockTwitsMessage(
                id=99,
                body="INFY mooning",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                sentiment="bullish",
            )
        ],
        bullish_count=1,
        bearish_count=0,
        neutral_count=0,
    )

    with (
        patch.object(sentiment_client, "_fetch_reddit", side_effect=ConnectionError("Reddit down")),
        patch.object(sentiment_client, "_fetch_stocktwits", new_callable=AsyncMock, return_value=fake_stocktwits),
        patch.object(sentiment_client, "_fetch_trends", new_callable=AsyncMock, return_value=TrendsSentiment(ticker="RELIANCE")),
    ):
        result = await sentiment_client.get_sentiment("RELIANCE", settings)

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.stocktwits.bullish_count == 1
