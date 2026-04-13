from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    MarketReport,
    NewsReport,
    SentimentAnalystReport,
)
from financialagents.agents.managers.schemas import Finding, ResearchBrief
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings



@pytest.fixture
def market_report() -> MarketReport:
    return MarketReport(
        rsi=55.0,
        rsi_interpretation="Neutral",
        macd_signal="bullish_crossover",
        trend_assessment="Mild uptrend",
        confidence=0.85,
        data_sources=["angelone"],
        notes="",
    )


@pytest.fixture
def fundamentals_report() -> FundamentalsAnalystReport:
    return FundamentalsAnalystReport(
        pe_ratio=22.5,
        pb_ratio=3.1,
        roce=18.0,
        debt_to_equity=0.4,
        fundamental_assessment="Healthy balance sheet",
        confidence=0.90,
        data_sources=["screener"],
        notes="",
    )


@pytest.fixture
def news_report() -> NewsReport:
    return NewsReport(
        article_count=5,
        sentiment_direction="positive",
        news_assessment="Positive news flow",
        confidence=0.70,
        data_sources=["moneycontrol"],
        notes="",
    )


@pytest.fixture
def sentiment_report() -> SentimentAnalystReport:
    return SentimentAnalystReport(
        reddit_score=0.6,
        stocktwits_score=0.5,
        combined_sentiment="bullish",
        sentiment_assessment="Moderate retail bullishness",
        confidence=0.60,
        data_sources=["reddit"],
        notes="",
    )


@pytest.fixture
def populated_state(
    market_report: MarketReport,
    fundamentals_report: FundamentalsAnalystReport,
    news_report: NewsReport,
    sentiment_report: SentimentAnalystReport,
) -> AgentState:
    return AgentState(
        ticker="RELIANCE",
        date="2024-01-15",
        market_report=market_report,
        fundamentals_report=fundamentals_report,
        news_report=news_report,
        sentiment_report=sentiment_report,
    )


@pytest.fixture
def mock_research_brief() -> ResearchBrief:
    return ResearchBrief(
        key_findings=[
            Finding(analyst="market", key_point="Mild uptrend", weight=0.25),
            Finding(analyst="fundamentals", key_point="Healthy balance sheet", weight=0.375),
            Finding(analyst="news", key_point="Positive news flow", weight=0.20),
            Finding(analyst="sentiment", key_point="Moderate retail bullishness", weight=0.175),
        ],
        consensus_view="Overall positive outlook for RELIANCE",
        conflicts=[],
        data_quality_summary="High quality data across all analysts",
        analyst_confidences={},
        overall_confidence=0.78,
        notes="All four analysts in agreement",
    )


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None)



class TestNormalizeWeights:
    def test_weights_sum_to_one(self):
        from financialagents.agents.managers.research_manager import normalize_weights

        raw = {"market": 1.0, "fundamentals": 1.5, "news": 0.8, "sentiment": 0.7}
        result = normalize_weights(raw)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_preserves_relative_ordering(self):
        from financialagents.agents.managers.research_manager import normalize_weights

        raw = {"market": 1.0, "fundamentals": 1.5, "news": 0.8, "sentiment": 0.7}
        result = normalize_weights(raw)
        assert result["fundamentals"] > result["market"] > result["news"] > result["sentiment"]

    def test_zero_total_returns_equal_weights(self):
        from financialagents.agents.managers.research_manager import normalize_weights

        raw = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = normalize_weights(raw)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert all(abs(v - 1.0 / 3) < 1e-9 for v in result.values())

    def test_negative_weights_clamped_to_zero(self):
        from financialagents.agents.managers.research_manager import normalize_weights

        raw = {"market": -1.0, "fundamentals": 2.0, "news": -0.5, "sentiment": 1.0}
        result = normalize_weights(raw)

        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result["market"] == 0.0
        assert result["news"] == 0.0
        assert abs(result["fundamentals"] - 2.0 / 3) < 1e-9
        assert abs(result["sentiment"] - 1.0 / 3) < 1e-9



class TestResearchManagerNode:
    def test_returns_research_summary_with_key_findings(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from financialagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        assert "research_summary" in result
        brief = result["research_summary"]
        assert isinstance(brief, ResearchBrief)
        assert len(brief.key_findings) > 0

    def test_populates_analyst_confidences_from_state(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from financialagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        brief = result["research_summary"]
        assert "market" in brief.analyst_confidences
        assert "fundamentals" in brief.analyst_confidences
        assert "news" in brief.analyst_confidences
        assert "sentiment" in brief.analyst_confidences
        assert abs(brief.analyst_confidences["market"] - 0.85) < 1e-6
        assert abs(brief.analyst_confidences["fundamentals"] - 0.90) < 1e-6
        assert abs(brief.analyst_confidences["news"] - 0.70) < 1e-6
        assert abs(brief.analyst_confidences["sentiment"] - 0.60) < 1e-6

    def test_prompt_includes_weight_emphasis(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
    ):
        from financialagents.agents.managers.research_manager import (
            build_research_manager_prompt,
            normalize_weights,
        )

        weights = {"market": 1.0, "fundamentals": 1.5, "news": 0.8, "sentiment": 0.7}
        norm_weights = normalize_weights(weights)

        messages = build_research_manager_prompt(
            market_report=populated_state.market_report,
            fundamentals_report=populated_state.fundamentals_report,
            news_report=populated_state.news_report,
            sentiment_report=populated_state.sentiment_report,
            weights=norm_weights,
        )

        full_text = " ".join(m.content for m in messages)
        assert "Fundamentals" in full_text or "fundamentals" in full_text
        assert "emphasis" in full_text.lower() or "weight" in full_text.lower() or "%" in full_text

    def test_retries_llm_on_failure(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from financialagents.agents.managers.research_manager import research_manager_node

        ainvoke_mock = AsyncMock(
            side_effect=[RuntimeError("LLM timeout"), mock_research_brief]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        assert "research_summary" in result
        assert ainvoke_mock.call_count == 2

    def test_handles_zero_confidence_report_in_prompt(
        self,
        fundamentals_report: FundamentalsAnalystReport,
        news_report: NewsReport,
        sentiment_report: SentimentAnalystReport,
    ):
        from financialagents.agents.managers.research_manager import (
            build_research_manager_prompt,
            normalize_weights,
        )

        zero_market_report = MarketReport(
            confidence=0.0,
            data_sources=[],
            notes="No OHLCV data",
        )

        weights = normalize_weights(
            {"market": 1.0, "fundamentals": 1.5, "news": 0.8, "sentiment": 0.7}
        )
        messages = build_research_manager_prompt(
            market_report=zero_market_report,
            fundamentals_report=fundamentals_report,
            news_report=news_report,
            sentiment_report=sentiment_report,
            weights=weights,
        )

        full_text = " ".join(m.content for m in messages)
        assert "NO DATA" in full_text or "no data" in full_text.lower() or "confidence: 0.0" in full_text.lower()

    def test_uses_deep_think_llm(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from financialagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            asyncio.run(research_manager_node(populated_state, config))

        mock_create.assert_called_once_with("deep_think", default_settings)

    def test_uses_analyst_weights_from_settings(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
    ):
        from financialagents.agents.managers.research_manager import research_manager_node

        settings = Settings(_env_file=None)
        settings.analyst_weights = {"market": 2.0, "fundamentals": 1.0, "news": 1.0, "sentiment": 1.0}

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": settings}}

        with patch(
            "financialagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        assert "research_summary" in result



class TestResearchBriefConfidenceValidation:

    @pytest.mark.parametrize("bad_value", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_research_brief_rejects_out_of_range_overall_confidence(self, bad_value: float):
        with pytest.raises(pydantic.ValidationError):
            ResearchBrief(overall_confidence=bad_value)

    @pytest.mark.parametrize("good_value", [0.0, 0.5, 1.0])
    def test_research_brief_accepts_valid_overall_confidence(self, good_value: float):
        brief = ResearchBrief(overall_confidence=good_value)
        assert brief.overall_confidence == good_value
