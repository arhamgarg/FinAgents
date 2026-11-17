from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.analysts.schemas import NewsReport
from finagents.config import Settings
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.interface import use_data_sources
from finagents.dataflows.news_schema import NewsArticle


@pytest.fixture
def agent_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="TCS", date="2024-12-31")


@pytest.fixture
def news_articles() -> list[NewsArticle]:
    return [
        NewsArticle(
            title="TCS Q3 results: Net profit rises 12% YoY",
            url="https://www.moneycontrol.com/news/tcs-q3-results",
            published_at=datetime(2024, 12, 15, 10, 0, tzinfo=UTC),
            source="moneycontrol",
            snippet=(
                "Tata Consultancy Services reports strong Q3 performance with IT services demand "
                "robust."
            ),
        ),
        NewsArticle(
            title="TCS bags multi-year deal with European bank",
            url="https://economictimes.indiatimes.com/tcs-deal",
            published_at=datetime(2024, 12, 10, 8, 30, tzinfo=UTC),
            source="economictimes",
            snippet="TCS secures a $500M multi-year digital transformation deal.",
        ),
        NewsArticle(
            title="IT sector outlook positive for 2025: Analysts",
            url="https://www.livemint.com/it-sector-2025",
            published_at=datetime(2024, 12, 8, 14, 0, tzinfo=UTC),
            source="livemint",
            snippet=(
                "Analysts expect Indian IT majors to benefit from global tech spending recovery."
            ),
        ),
    ]


@pytest.fixture
def valid_news_report() -> NewsReport:
    return NewsReport(
        article_count=3,
        sentiment_direction="positive",
        top_headlines=[
            "TCS Q3 results: Net profit rises 12% YoY",
            "TCS bags multi-year deal with European bank",
            "IT sector outlook positive for 2025: Analysts",
        ],
        key_themes=["quarterly results", "deal wins", "IT sector outlook"],
        news_assessment=(
            "News sentiment for TCS is positive. Strong Q3 results and a major deal win "
            "from a European bank signal continued revenue growth. IT sector tailwinds "
            "support a bullish outlook for 2025."
        ),
        confidence=0.8,
        data_sources=["moneycontrol"],
        notes="3 articles from Indian financial publications.",
    )


class TestNewsAnalystNode:
    @pytest.mark.asyncio
    async def test_returns_news_report(
        self,
        agent_state,
        news_articles: list[NewsArticle],
        valid_news_report: NewsReport,
    ):
        data_result = DataResult(status="ok", data=news_articles, source="moneycontrol")

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=valid_news_report)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with (
            use_data_sources(news=AsyncMock(return_value=data_result)),
            patch(
                "finagents.agents.analysts.news_analyst.create_llm_client",
                return_value=mock_llm,
            ),
        ):
            from finagents.agents.analysts.news_analyst import news_analyst_node

            result = await news_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "news_report" in result
        report = result["news_report"]
        assert isinstance(report, NewsReport)
        assert report.article_count == 3
        assert report.sentiment_direction == "positive"
        assert report.confidence > 0

    @pytest.mark.asyncio
    async def test_handles_empty_articles_with_zero_confidence(self, agent_state):
        data_result = DataResult(status="empty", data=None)

        with use_data_sources(news=AsyncMock(return_value=data_result)):
            from finagents.agents.analysts.news_analyst import news_analyst_node

            result = await news_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "news_report" in result
        report = result["news_report"]
        assert isinstance(report, NewsReport)
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_empty_article_list_with_zero_confidence(self, agent_state):
        data_result = DataResult(status="ok", data=[], source="moneycontrol")

        with use_data_sources(news=AsyncMock(return_value=data_result)):
            from finagents.agents.analysts.news_analyst import news_analyst_node

            result = await news_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "news_report" in result
        assert result["news_report"].confidence == 0.0

    @pytest.mark.asyncio
    async def test_retries_llm_once_on_failure(
        self,
        agent_state,
        news_articles: list[NewsArticle],
        valid_news_report: NewsReport,
    ):
        data_result = DataResult(status="ok", data=news_articles, source="moneycontrol")

        mock_ainvoke = AsyncMock(side_effect=[RuntimeError("LLM timeout"), valid_news_report])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = mock_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with (
            use_data_sources(news=AsyncMock(return_value=data_result)),
            patch(
                "finagents.agents.analysts.news_analyst.create_llm_client",
                return_value=mock_llm,
            ),
        ):
            from finagents.agents.analysts.news_analyst import news_analyst_node

            result = await news_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "news_report" in result
        assert isinstance(result["news_report"], NewsReport)
        assert mock_ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_article_count_set_from_data(
        self,
        agent_state,
        news_articles: list[NewsArticle],
    ):
        data_result = DataResult(status="ok", data=news_articles, source="moneycontrol")

        report_no_count = NewsReport(
            article_count=0,
            sentiment_direction="positive",
            news_assessment="Good news flow.",
            confidence=0.7,
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=report_no_count)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with (
            use_data_sources(news=AsyncMock(return_value=data_result)),
            patch(
                "finagents.agents.analysts.news_analyst.create_llm_client",
                return_value=mock_llm,
            ),
        ):
            from finagents.agents.analysts.news_analyst import news_analyst_node

            result = await news_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert result["news_report"].article_count == 3
