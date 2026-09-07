from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.analysts.schemas import NewsReport
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.news_schema import NewsArticle




@pytest.fixture
def agent_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="TESTCO", date="2024-12-31")


def _make_articles(n: int) -> list[NewsArticle]:
    now = datetime(2024, 12, 15, 10, 0, tzinfo=timezone.utc)
    return [
        NewsArticle(
            title=f"TESTCO headline {i}",
            url=f"https://example.com/{i}",
            published_at=now,
            source="moneycontrol",
            snippet=f"Snippet {i}.",
        )
        for i in range(n)
    ]


async def _run_node_with_mocks(
    agent_state, articles: list[NewsArticle], llm_emitted_confidence: float
):
    mock_report = NewsReport(
        article_count=999,
        sentiment_direction="positive",
        confidence=llm_emitted_confidence,
    )
    if len(articles) == 0:
        data_result = DataResult(status="empty", data=None, error="no news")
        mock_llm = MagicMock()
    else:
        data_result = DataResult(status="ok", data=articles, source="moneycontrol")
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_report)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
    with (
        patch(
            "finagents.agents.analysts.news_analyst.get_news",
            new_callable=AsyncMock,
            return_value=data_result,
        ),
        patch(
            "finagents.agents.analysts.news_analyst.create_llm_client",
            return_value=mock_llm,
        ),
    ):
        from finagents.agents.analysts.news_analyst import news_analyst_node

        return await news_analyst_node(agent_state, {"configurable": {}})




class TestNewsAnalystConfidence:
    @pytest.mark.asyncio
    async def test_zero_articles_yields_0_0(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(0), llm_emitted_confidence=0.0
        )
        report = result["news_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_one_article_yields_0_3(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(1), llm_emitted_confidence=0.9
        )
        report = result["news_report"]
        assert report.confidence == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_two_articles_yields_0_3(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(2), llm_emitted_confidence=0.95
        )
        report = result["news_report"]
        assert report.confidence == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_three_articles_yields_0_6(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(3), llm_emitted_confidence=0.05
        )
        report = result["news_report"]
        assert report.confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_five_articles_yields_0_6(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(5), llm_emitted_confidence=0.99
        )
        report = result["news_report"]
        assert report.confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_six_plus_articles_yields_1_0(self, agent_state):
        result = await _run_node_with_mocks(
            agent_state, _make_articles(6), llm_emitted_confidence=0.1
        )
        report = result["news_report"]
        assert report.confidence == pytest.approx(1.0)
