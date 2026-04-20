from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.analysts.schemas import SentimentAnalystReport
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.sentiment_schema import (
    RedditPost,
    RedditSentiment,
    SentimentReport,
    StockTwitsMessage,
    StockTwitsSentiment,
    TrendsSentiment,
)




@pytest.fixture
def agent_state():
    from financialagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="INFY", date="2024-12-31")


@pytest.fixture
def sentiment_report() -> SentimentReport:
    now = datetime(2024, 12, 15, 10, 0, tzinfo=timezone.utc)
    return SentimentReport(
        ticker="INFY",
        reddit=RedditSentiment(
            posts=[
                RedditPost(
                    title="INFY looking bullish after deal wins",
                    score=120,
                    url="https://reddit.com/r/IndianStreetBets/comments/abc",
                    created_utc=now,
                    subreddit="IndianStreetBets",
                    selftext_preview="Great deal wins this quarter",
                ),
                RedditPost(
                    title="Infosys guidance raised — bullish!",
                    score=85,
                    url="https://reddit.com/r/DalalStreetTalks/comments/xyz",
                    created_utc=now,
                    subreddit="DalalStreetTalks",
                    selftext_preview="Management raised guidance",
                ),
            ],
            post_count=2,
            avg_score=102.5,
        ),
        stocktwits=StockTwitsSentiment(
            messages=[
                StockTwitsMessage(
                    id=1001,
                    body="$INFY strong buy here",
                    created_at=now,
                    sentiment="bullish",
                ),
                StockTwitsMessage(
                    id=1002,
                    body="$INFY just bought more",
                    created_at=now,
                    sentiment="bullish",
                ),
                StockTwitsMessage(
                    id=1003,
                    body="$INFY watching closely",
                    created_at=now,
                    sentiment=None,
                ),
            ],
            bullish_count=2,
            bearish_count=0,
            neutral_count=1,
        ),
        trends=TrendsSentiment(
            ticker="INFY",
            interest_over_time=[],
            peak_interest=72,
            avg_interest=58.5,
        ),
        sources_status={"reddit": "ok", "stocktwits": "ok", "trends": "ok"},
    )


@pytest.fixture
def valid_sentiment_analyst_report() -> SentimentAnalystReport:
    return SentimentAnalystReport(
        reddit_score=0.75,
        stocktwits_score=0.80,
        combined_sentiment="bullish",
        post_count=5,
        sentiment_assessment=(
            "Retail sentiment for INFY is bullish. Reddit posts in r/IndianStreetBets and "
            "r/DalalStreetTalks show strong positive bias with deal win mentions. "
            "StockTwits shows 2 bullish vs 0 bearish messages. Overall sentiment is bullish."
        ),
        confidence=0.75,
        data_sources=["reddit", "stocktwits", "trends"],
        notes="Combined Reddit + StockTwits + Trends sentiment for INFY.",
    )




class TestSentimentAnalystNode:
    @pytest.mark.asyncio
    async def test_returns_sentiment_report(
        self,
        agent_state,
        sentiment_report: SentimentReport,
        valid_sentiment_analyst_report: SentimentAnalystReport,
    ):
        data_result = DataResult(
            status="ok", data=sentiment_report, source="reddit"
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value=valid_sentiment_analyst_report
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.sentiment_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        assert "sentiment_report" in result
        report = result["sentiment_report"]
        assert isinstance(report, SentimentAnalystReport)
        assert report.combined_sentiment == "bullish"
        assert report.confidence > 0

    @pytest.mark.asyncio
    async def test_handles_empty_sentiment_with_zero_confidence(self, agent_state):
        data_result = DataResult(status="empty", data=None)

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        assert "sentiment_report" in result
        report = result["sentiment_report"]
        assert isinstance(report, SentimentAnalystReport)
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_error_status_with_zero_confidence(self, agent_state):
        data_result = DataResult(
            status="error", data=None, error="Reddit API rate limit exceeded"
        )

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        assert "sentiment_report" in result
        assert result["sentiment_report"].confidence == 0.0

    @pytest.mark.asyncio
    async def test_retries_llm_once_on_failure(
        self,
        agent_state,
        sentiment_report: SentimentReport,
        valid_sentiment_analyst_report: SentimentAnalystReport,
    ):
        data_result = DataResult(
            status="ok", data=sentiment_report, source="reddit"
        )

        mock_ainvoke = AsyncMock(
            side_effect=[RuntimeError("LLM timeout"), valid_sentiment_analyst_report]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = mock_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.sentiment_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        assert "sentiment_report" in result
        assert isinstance(result["sentiment_report"], SentimentAnalystReport)
        assert mock_ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_post_count_set_from_reddit_and_stocktwits(
        self,
        agent_state,
        sentiment_report: SentimentReport,
    ):
        data_result = DataResult(
            status="ok", data=sentiment_report, source="reddit"
        )

        report_no_count = SentimentAnalystReport(
            combined_sentiment="bullish",
            post_count=0,
            sentiment_assessment="Bullish sentiment.",
            confidence=0.7,
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=report_no_count)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.sentiment_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        assert result["sentiment_report"].post_count == 5

    @pytest.mark.asyncio
    async def test_data_sources_from_sources_status(
        self,
        agent_state,
        sentiment_report: SentimentReport,
    ):
        data_result = DataResult(
            status="ok", data=sentiment_report, source="reddit"
        )

        report_no_sources = SentimentAnalystReport(
            combined_sentiment="bullish",
            sentiment_assessment="Bullish.",
            confidence=0.7,
            data_sources=[],
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=report_no_sources)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.sentiment_analyst.get_sentiment",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.sentiment_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.sentiment_analyst import (
                sentiment_analyst_node,
            )

            result = await sentiment_analyst_node(agent_state, {"configurable": {}})

        sources = result["sentiment_report"].data_sources
        assert "reddit" in sources
        assert "stocktwits" in sources
        assert "trends" in sources
