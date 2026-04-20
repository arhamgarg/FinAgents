from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.analysts.schemas import TechnicalReport
from financialagents.agents.managers.schemas import Finding, ResearchBrief
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.agents.utils.agent_states import AgentState, InvestDebateState
from financialagents.config import Settings




@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def mock_technical_report() -> TechnicalReport:
    return TechnicalReport(
        rsi=55.0,
        rsi_interpretation="neutral",
        macd_signal="bullish_crossover",
        trend_assessment="Uptrend",
        current_price=2450.75,
        atr_14=50.0,
        circuit_band_pct=0.20,
        t2t_flag=False,
        confidence=0.85,
        data_sources=["angelone"],
    )


@pytest.fixture
def mock_research_brief() -> ResearchBrief:
    return ResearchBrief(
        key_findings=[
            Finding(analyst="technical", key_point="Uptrend", weight=0.25)
        ],
        consensus_view="Positive outlook",
        overall_confidence=0.76,
        analyst_confidences={
            "technical": 0.85,
            "fundamentals": 0.90,
            "news": 0.70,
        },
    )


@pytest.fixture
def mock_trading_signal() -> TradingSignal:
    return TradingSignal(
        action="BUY",
        conviction=0.75,
        entry_price=1500.0,
        stop_loss=2400.0,
        target_1=2500.0,
        rationale="seed rationale",
    )


@pytest.fixture
def populated_state(
    mock_technical_report: TechnicalReport,
    mock_research_brief: ResearchBrief,
) -> AgentState:
    debate = InvestDebateState(
        max_rounds=3,
        research_brief_snapshot="bullish",
        investment_plan="BUY directional (bull thesis outweighed bear)",
    )
    return AgentState(
        ticker="TRENT",
        date="2026-04-17",
        technical_report=mock_technical_report,
        research_summary=mock_research_brief,
        investment_debate_state=debate,
    )




class TestTraderNode:
    def test_returns_trader_signal(
        self,
        populated_state: AgentState,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(populated_state, config))

        assert "trader_signal" in result
        assert isinstance(result["trader_signal"], TradingSignal)

    def test_uses_deep_think_tier(
        self,
        populated_state: AgentState,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            asyncio.run(trader_node(populated_state, config))

        mock_create.assert_called_once_with("deep_think", default_settings)

    def test_price_grounding_overrides_llm_entry_price(
        self,
        populated_state: AgentState,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(populated_state, config))

        signal = result["trader_signal"]
        assert signal.entry_price == pytest.approx(2450.75)

    def test_analyst_weight_applied_matches_settings(
        self,
        populated_state: AgentState,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(populated_state, config))

        signal = result["trader_signal"]
        assert signal.analyst_weight_applied == dict(default_settings.analyst_weights)

    def test_retry_on_failure(
        self,
        populated_state: AgentState,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        ainvoke_mock = AsyncMock(
            side_effect=[RuntimeError("timeout"), mock_trading_signal]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(populated_state, config))

        assert ainvoke_mock.call_count == 2
        assert result["trader_signal"] is not None
        assert result["trader_signal"].action == "BUY"

    def test_double_failure_degrades_to_hold(
        self,
        populated_state: AgentState,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        ainvoke_mock = AsyncMock(
            side_effect=[RuntimeError("first error"), RuntimeError("second error")]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(populated_state, config))

        signal = result["trader_signal"]
        assert signal.action == "HOLD"
        assert signal.conviction == 0.0
        assert "LLM invocation failed" in signal.rationale

    def test_current_price_none_forces_hold(
        self,
        mock_research_brief: ResearchBrief,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        tech = TechnicalReport(
            current_price=None,
            atr_14=50.0,
            circuit_band_pct=0.20,
            t2t_flag=False,
            confidence=0.5,
        )
        state = AgentState(
            ticker="X",
            date="2026-04-17",
            technical_report=tech,
            research_summary=mock_research_brief,
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(state, config))

        signal = result["trader_signal"]
        assert signal.action == "HOLD"

    def test_circuit_band_none_forces_hold_with_canonical_note(
        self,
        mock_research_brief: ResearchBrief,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node
        from financialagents.agents.trader.validators import (
            CIRCUIT_BAND_UNAVAILABLE_NOTE,
        )

        tech = TechnicalReport(
            current_price=2450.75,
            atr_14=50.0,
            circuit_band_pct=None,
            t2t_flag=False,
            confidence=0.5,
        )
        state = AgentState(
            ticker="X",
            date="2026-04-17",
            technical_report=tech,
            research_summary=mock_research_brief,
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(state, config))

        signal = result["trader_signal"]
        assert signal.action == "HOLD"
        assert signal.circuit_band_note == CIRCUIT_BAND_UNAVAILABLE_NOTE

    def test_t2t_flag_copied_from_technical_report(
        self,
        mock_research_brief: ResearchBrief,
        mock_trading_signal: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.trader.trader import trader_node

        tech = TechnicalReport(
            current_price=2450.75,
            atr_14=50.0,
            circuit_band_pct=0.20,
            t2t_flag=True,
            confidence=0.8,
        )
        state = AgentState(
            ticker="X",
            date="2026-04-17",
            technical_report=tech,
            research_summary=mock_research_brief,
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_trading_signal)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = {"configurable": {"settings": default_settings}}

        with patch(
            "financialagents.agents.trader.trader.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(trader_node(state, config))

        assert result["trader_signal"].t2t_flag is True




def test_pipe12_no_forbidden_analyst_reads() -> None:
    import re
    from pathlib import Path

    source_path = (
        Path(__file__).parent.parent.parent.parent
        / "financialagents/agents/trader/trader.py"
    )
    source = source_path.read_text()
    forbidden = [
        "state.fundamentals_report",
        "state.news_report",
        "technical_report.rsi_interpretation",
        "technical_report.macd_signal",
        "technical_report.trend_assessment",
        "technical_report.bollinger_position",
        "technical_report.ma_crossover",
    ]
    for pattern in forbidden:
        assert not re.search(re.escape(pattern), source), (
            f" violation: trader.py reads forbidden field: {pattern}"
        )


test_pipe12 = test_pipe12_no_forbidden_analyst_reads
