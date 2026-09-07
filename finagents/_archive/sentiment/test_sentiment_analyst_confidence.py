from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.analysts.schemas import SentimentAnalystReport
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.sentiment_schema import (
    RedditPost,
    RedditSentiment,
    SentimentReport,
    StockTwitsMessage,
    StockTwitsSentiment,
    TrendsSentiment,
)




def _make_sentiment_data(*, reddit_posts: int, stocktwits_msgs: int) -> SentimentReport:
    now = datetime(2024, 12, 15, 10, 0, tzinfo=timezone.utc)
    return SentimentReport(
        ticker="TESTCO",
        reddit=RedditSentiment(
            posts=[
                RedditPost(
                    title=f"p{i}",
                    score=10,
                    url=f"https://r/{i}",
                    created_utc=now,
                    subreddit="IndianStreetBets",
                    selftext_preview="",
                )
                for i in range(reddit_posts)
            ],
            post_count=reddit_posts,
            avg_score=10.0,
        ),
        stocktwits=StockTwitsSentiment(
            messages=[
                StockTwitsMessage(
                    id=1000 + i,
                    body=f"m{i}",
                    created_at=now,
                    sentiment="bullish",
                )
                for i in range(stocktwits_msgs)
            ],
            bullish_count=stocktwits_msgs,
            bearish_count=0,
            neutral_count=0,
        ),
        trends=TrendsSentiment(
            ticker="TESTCO",
            interest_over_time=[],
            peak_interest=50,
            avg_interest=30.0,
        ),
        sources_status={"reddit": "ok", "stocktwits": "ok", "trends": "ok"},
    )


@pytest.fixture
def agent_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="TESTCO", date="2024-12-31")


async def _run_node_with_mocks(
    agent_state,
    sentiment_data: SentimentReport | None,
    llm_emitted_confidence: float,
):
    mock_report = SentimentAnalystReport(
        reddit_score=0.5,
        stocktwits_score=0.5,
        combined_sentiment="bullish",
        post_count=999,
        confidence=llm_emitted_confidence,
    )
    if sentiment_data is None:
        data_result = DataResult(status="empty", data=None, error="no sentiment")
        mock_llm = MagicMock()
    else:
        data_result = DataResult(status="ok", data=sentiment_data, source="reddit")
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_report)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
    with (
        patch(
            "finagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ),
        patch(
            "finagents.agents.analysts.sentiment_analyst.create_llm_client",
            return_value=mock_llm,
        ),
    ):
        from finagents.agents.analysts.sentiment_analyst import (
            sentiment_analyst_node,
        )

        return await sentiment_analyst_node(agent_state, {"configurable": {}})




class TestSentimentAnalystConfidence:
    @pytest.mark.asyncio
    async def test_empty_posts_early_return_yields_0_0(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, sentiment_data=None, llm_emitted_confidence=0.8
        )
        report = result["sentiment_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_zero_post_count_yields_0_0(self, agent_state):
        sentiment_data = _make_sentiment_data(reddit_posts=0, stocktwits_msgs=0)
        result = await _run_node_with_mocks(
            agent_state, sentiment_data=sentiment_data, llm_emitted_confidence=0.8
        )
        report = result["sentiment_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_post_count_cap_dominates(self, agent_state):
        sentiment_data = _make_sentiment_data(reddit_posts=1, stocktwits_msgs=1)
        result = await _run_node_with_mocks(
            agent_state, sentiment_data=sentiment_data, llm_emitted_confidence=0.9
        )
        report = result["sentiment_report"]
        assert report.confidence == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_llm_cap_dominates(self, agent_state):
        sentiment_data = _make_sentiment_data(reddit_posts=10, stocktwits_msgs=10)
        result = await _run_node_with_mocks(
            agent_state, sentiment_data=sentiment_data, llm_emitted_confidence=0.5
        )
        report = result["sentiment_report"]
        assert report.confidence == pytest.approx(0.5)
