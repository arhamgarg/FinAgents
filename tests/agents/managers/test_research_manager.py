from __future__ import annotations

from langchain_core.runnables import RunnableConfig

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest

from finagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    TechnicalReport,
    NewsReport,
)
from finagents.agents.managers.schemas import Finding, ResearchBrief
from finagents.agents.utils.agent_states import AgentState
from finagents.config import Settings




@pytest.fixture
def technical_report() -> TechnicalReport:
    return TechnicalReport(
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
def populated_state(
    technical_report: TechnicalReport,
    fundamentals_report: FundamentalsAnalystReport,
    news_report: NewsReport,
) -> AgentState:
    return AgentState(
        ticker="RELIANCE",
        date="2024-01-15",
        technical_report=technical_report,
        fundamentals_report=fundamentals_report,
        news_report=news_report,
    )


@pytest.fixture
def mock_research_brief() -> ResearchBrief:
    return ResearchBrief(
        key_findings=[
            Finding(analyst="technical", key_point="Mild uptrend", weight=0.303),
            Finding(analyst="fundamentals", key_point="Healthy balance sheet", weight=0.455),
            Finding(analyst="news", key_point="Positive news flow", weight=0.242),
        ],
        consensus_view="Overall positive outlook for RELIANCE",
        conflicts=[],
        data_quality_summary="High quality data across all analysts",
        analyst_confidences={},
        overall_confidence=0.78,
        notes="All three analysts in agreement",
    )


@pytest.fixture
def default_settings() -> Settings:
    return Settings(openrouter_api_key="fake")




class TestNormalizeWeights:
    def test_weights_sum_to_one(self):
        from finagents.agents.managers.research_manager import normalize_weights

        raw = {"technical": 1.0, "fundamentals": 1.5, "news": 0.8}
        result = normalize_weights(raw)
        assert abs(result["technical"] - 0.303) < 1e-3
        assert abs(result["fundamentals"] - 0.455) < 1e-3
        assert abs(result["news"] - 0.242) < 1e-3
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_preserves_relative_ordering(self):
        from finagents.agents.managers.research_manager import normalize_weights

        raw = {"technical": 1.0, "fundamentals": 1.5, "news": 0.8}
        result = normalize_weights(raw)
        assert result["fundamentals"] > result["technical"] > result["news"]

    def test_zero_total_returns_equal_weights(self):
        from finagents.agents.managers.research_manager import normalize_weights

        raw = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = normalize_weights(raw)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert all(abs(v - 1.0 / 3) < 1e-9 for v in result.values())

    def test_negative_weights_clamped_to_zero(self):
        from finagents.agents.managers.research_manager import normalize_weights

        raw = {"technical": -1.0, "fundamentals": 2.0, "news": 1.0}
        result = normalize_weights(raw)

        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result["technical"] == 0.0
        assert abs(result["fundamentals"] - 2.0 / 3) < 1e-9
        assert abs(result["news"] - 1.0 / 3) < 1e-9




class TestResearchManagerNode:
    def test_returns_research_summary_with_key_findings(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from finagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.managers.research_manager.create_llm_client",
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
        from finagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        brief = result["research_summary"]
        assert "technical" in brief.analyst_confidences
        assert "fundamentals" in brief.analyst_confidences
        assert "news" in brief.analyst_confidences
        assert "sentiment" not in brief.analyst_confidences
        assert abs(brief.analyst_confidences["technical"] - 0.85) < 1e-6
        assert abs(brief.analyst_confidences["fundamentals"] - 0.90) < 1e-6
        assert abs(brief.analyst_confidences["news"] - 0.70) < 1e-6

    def test_prompt_includes_weight_emphasis(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
    ):
        from finagents.agents.managers.research_manager import (
            build_research_manager_prompt,
            normalize_weights,
        )

        weights = {"technical": 1.0, "fundamentals": 1.5, "news": 0.8}
        norm_weights = normalize_weights(weights)

        messages = build_research_manager_prompt(
            technical_report=populated_state.technical_report,
            fundamentals_report=populated_state.fundamentals_report,
            news_report=populated_state.news_report,
            weights=norm_weights,
        )

        full_text = " ".join(m.text for m in messages)
        assert "Fundamentals" in full_text or "fundamentals" in full_text
        assert "emphasis" in full_text.lower() or "weight" in full_text.lower() or "%" in full_text

    def test_retries_llm_on_failure(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from finagents.agents.managers.research_manager import research_manager_node

        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("LLM timeout"), mock_research_brief])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(research_manager_node(populated_state, config))

        assert "research_summary" in result
        assert ainvoke_mock.call_count == 2

    def test_handles_zero_confidence_report_in_prompt(
        self,
        fundamentals_report: FundamentalsAnalystReport,
        news_report: NewsReport,
    ):
        from finagents.agents.managers.research_manager import (
            build_research_manager_prompt,
            normalize_weights,
        )

        zero_technical_report = TechnicalReport(
            confidence=0.0,
            data_sources=[],
            notes="No OHLCV data",
        )

        weights = normalize_weights({"technical": 1.0, "fundamentals": 1.5, "news": 0.8})
        messages = build_research_manager_prompt(
            technical_report=zero_technical_report,
            fundamentals_report=fundamentals_report,
            news_report=news_report,
            weights=weights,
        )

        full_text = " ".join(m.text for m in messages)
        assert (
            "NO DATA" in full_text
            or "no data" in full_text.lower()
            or "confidence: 0.0" in full_text.lower()
        )

    def test_uses_deep_think_llm(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
        default_settings: Settings,
    ):
        from finagents.agents.managers.research_manager import research_manager_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.managers.research_manager.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            asyncio.run(research_manager_node(populated_state, config))

        mock_create.assert_called_once_with(
            "deep_think", default_settings, call_site="research_manager"
        )

    def test_uses_analyst_weights_from_settings(
        self,
        populated_state: AgentState,
        mock_research_brief: ResearchBrief,
    ):
        from finagents.agents.managers.research_manager import research_manager_node

        settings = Settings(openrouter_api_key="fake")
        settings.analyst_weights = {"technical": 2.0, "fundamentals": 1.0, "news": 1.0}

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_research_brief)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config: RunnableConfig = {"configurable": {"settings": settings}}

        with patch(
            "finagents.agents.managers.research_manager.create_llm_client",
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
