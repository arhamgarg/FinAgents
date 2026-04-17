from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    TechnicalReport,
    NewsReport,
    SentimentAnalystReport,
)
from financialagents.agents.managers.schemas import Finding, ResearchBrief
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings



@pytest.fixture
def mock_technical_report() -> TechnicalReport:
    return TechnicalReport(
        rsi=55.0,
        trend_assessment="Uptrend",
        confidence=0.85,
        data_sources=["angelone"],
    )


@pytest.fixture
def mock_fundamentals_report() -> FundamentalsAnalystReport:
    return FundamentalsAnalystReport(
        pe_ratio=22.0,
        fundamental_assessment="Healthy",
        confidence=0.90,
        data_sources=["screener"],
    )


@pytest.fixture
def mock_news_report() -> NewsReport:
    return NewsReport(
        article_count=3,
        sentiment_direction="positive",
        news_assessment="Positive news",
        confidence=0.70,
        data_sources=["moneycontrol"],
    )


@pytest.fixture
def mock_sentiment_report() -> SentimentAnalystReport:
    return SentimentAnalystReport(
        reddit_score=0.6,
        combined_sentiment="bullish",
        sentiment_assessment="Bullish retail",
        confidence=0.60,
        data_sources=["reddit"],
    )


@pytest.fixture
def mock_research_brief() -> ResearchBrief:
    return ResearchBrief(
        key_findings=[Finding(analyst="technical", key_point="Uptrend", weight=0.25)],
        consensus_view="Positive outlook",
        overall_confidence=0.76,
        analyst_confidences={
            "technical": 0.85,
            "fundamentals": 0.90,
            "news": 0.70,
            "sentiment": 0.60,
        },
    )


@pytest.fixture
def full_state(
    mock_technical_report,
    mock_fundamentals_report,
    mock_news_report,
    mock_sentiment_report,
) -> AgentState:
    return AgentState(
        ticker="TCS",
        date="2024-01-15",
        technical_report=mock_technical_report,
        fundamentals_report=mock_fundamentals_report,
        news_report=mock_news_report,
        sentiment_report=mock_sentiment_report,
    )



class TestBuildAnalysisGraph:
    def test_graph_compiles(self):
        from financialagents.graph.setup import build_analysis_graph

        graph = build_analysis_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from financialagents.graph.setup import build_analysis_graph

        graph = build_analysis_graph()
        node_names = list(graph.nodes.keys())
        expected_nodes = [
            "propagation_node",
            "technical_analyst_node",
            "fundamentals_analyst_node",
            "news_analyst_node",
            "sentiment_analyst_node",
            "validation_node",
            "research_manager_node",
        ]
        for node_name in expected_nodes:
            assert node_name in node_names, f"Node '{node_name}' not found in graph"



class TestValidationNode:
    def test_validation_fails_on_missing_report(self):
        from financialagents.graph.setup import validation_node

        state = AgentState(
            ticker="TCS",
            date="2024-01-15",
            technical_report=None,
            fundamentals_report=FundamentalsAnalystReport(confidence=0.9),
            news_report=NewsReport(confidence=0.7),
            sentiment_report=SentimentAnalystReport(confidence=0.6),
        )
        with pytest.raises(ValueError, match="technical_report"):
            validation_node(state)

    def test_validation_fails_listing_all_missing(self):
        from financialagents.graph.setup import validation_node

        state = AgentState(ticker="TCS", date="2024-01-15")
        with pytest.raises(ValueError) as exc_info:
            validation_node(state)
        error_msg = str(exc_info.value)
        assert "technical_report" in error_msg
        assert "fundamentals_report" in error_msg
        assert "news_report" in error_msg
        assert "sentiment_report" in error_msg

    def test_validation_passes_when_all_reports_present(self, full_state: AgentState):
        from financialagents.graph.setup import validation_node

        result = validation_node(full_state)
        assert result == {}



class TestPropagationNode:
    def test_propagation_node_returns_empty_dict(self):
        from financialagents.graph.setup import propagation_node

        state = AgentState(ticker="RELIANCE", date="2024-01-15")
        result = propagation_node(state)
        assert result == {}



class TestGraphIntegration:
    def test_full_graph_with_mocked_nodes(
        self,
        mock_technical_report,
        mock_fundamentals_report,
        mock_news_report,
        mock_sentiment_report,
        mock_research_brief,
    ):
        from financialagents.graph.setup import build_analysis_graph

        async def fake_technical(state, config):
            return {"technical_report": mock_technical_report}

        async def fake_fundamentals(state, config):
            return {"fundamentals_report": mock_fundamentals_report}

        async def fake_news(state, config):
            return {"news_report": mock_news_report}

        async def fake_sentiment(state, config):
            return {"sentiment_report": mock_sentiment_report}

        async def fake_research_manager(state, config):
            return {"research_summary": mock_research_brief}

        with (
            patch("financialagents.agents.analysts.technical_analyst.technical_analyst_node", fake_technical),
            patch("financialagents.agents.analysts.fundamentals_analyst.fundamentals_analyst_node", fake_fundamentals),
            patch("financialagents.agents.analysts.news_analyst.news_analyst_node", fake_news),
            patch("financialagents.agents.analysts.sentiment_analyst.sentiment_analyst_node", fake_sentiment),
            patch("financialagents.agents.managers.research_manager.research_manager_node", fake_research_manager),
        ):
            from importlib import reload
            import financialagents.graph.setup as setup_mod
            reload(setup_mod)
            graph = setup_mod.build_analysis_graph()

            initial_state = AgentState(ticker="TCS", date="2024-01-15")
            settings = Settings(_env_file=None)
            config = {"configurable": {"settings": settings}}
            result = asyncio.run(graph.ainvoke(initial_state, config=config))

        if isinstance(result, dict):
            state = AgentState.model_validate(result)
            assert state.technical_report is not None
            assert state.research_summary is not None
        else:
            assert result.technical_report is not None
            assert result.research_summary is not None



class TestFinancialAgentsGraph:
    def test_propagate_returns_agent_state(
        self,
        mock_technical_report,
        mock_fundamentals_report,
        mock_news_report,
        mock_sentiment_report,
        mock_research_brief,
    ):
        from financialagents.graph.trading_graph import FinancialAgentsGraph

        async def fake_ainvoke(state, config=None):
            return {
                "ticker": state.ticker,
                "date": state.date,
                "technical_report": mock_technical_report,
                "fundamentals_report": mock_fundamentals_report,
                "news_report": mock_news_report,
                "sentiment_report": mock_sentiment_report,
                "research_summary": mock_research_brief,
                "messages": [],
                "selected_tickers": [],
            }

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "financialagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa_graph = FinancialAgentsGraph()
            settings = Settings(_env_file=None)
            result = fa_graph.propagate("TCS", "2024-01-15", settings=settings)

        assert isinstance(result, AgentState)
        assert result.research_summary is not None
        assert result.ticker == "TCS"

    def test_propagate_creates_default_settings_when_none(
        self,
        mock_research_brief,
    ):
        from financialagents.graph.trading_graph import FinancialAgentsGraph

        async def fake_ainvoke(state, config=None):
            return {
                "ticker": state.ticker,
                "date": state.date,
                "technical_report": None,
                "fundamentals_report": None,
                "news_report": None,
                "sentiment_report": None,
                "research_summary": mock_research_brief,
                "messages": [],
                "selected_tickers": [],
            }

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "financialagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa_graph = FinancialAgentsGraph()
            result = fa_graph.propagate("INFY", "2024-01-15", settings=None)

        assert isinstance(result, AgentState)
        assert result.ticker == "INFY"
