from __future__ import annotations

import pytest
from pydantic import ValidationError

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    TechnicalReport,
    NewsReport,
    SentimentAnalystReport,
)
from financialagents.agents.managers.schemas import ResearchBrief
from financialagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class TestAgentStateValidation:

    def test_minimal_valid_state(self) -> None:
        state = AgentState(ticker="RELIANCE", date="2026-04-07")
        assert state.ticker == "RELIANCE"
        assert state.date == "2026-04-07"
        assert state.technical_report is None
        assert state.fundamentals_report is None
        assert state.final_trade_decision is None
        assert state.messages == []

    def test_empty_state_valid(self) -> None:
        state = AgentState()
        assert state.ticker == ""
        assert state.date == ""

    def test_full_state_valid(self) -> None:
        state = AgentState(
            ticker="TCS",
            date="2026-04-07",
            sector="IT",
            selected_tickers=["TCS", "INFY", "WIPRO"],
            technical_report=TechnicalReport(rsi=65.0, macd_signal="bullish_crossover", confidence=0.8),
            news_report=NewsReport(article_count=3, sentiment_direction="positive", confidence=0.7),
            sentiment_report=SentimentAnalystReport(combined_sentiment="bullish", confidence=0.6),
            fundamentals_report=FundamentalsAnalystReport(pe_ratio=28.0, roce=45.0, confidence=0.9),
            research_summary=ResearchBrief(consensus_view="Overall bullish outlook"),
            trader_investment_plan="Buy at 3800, SL 3700",
            final_trade_decision={"signal": "BUY", "confidence": 0.85},
        )
        assert state.sector == "IT"
        assert len(state.selected_tickers) == 3
        assert state.final_trade_decision["signal"] == "BUY"
        assert isinstance(state.technical_report, TechnicalReport)
        assert state.technical_report.rsi == pytest.approx(65.0)

    def test_rejects_invalid_ticker_type(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentState(ticker=12345)  # type: ignore[arg-type]
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("ticker",) for e in errors)

    def test_selected_tickers_must_be_list(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentState(selected_tickers="not-a-list")  # type: ignore[arg-type]
        errors = exc_info.value.errors()
        assert any("selected_tickers" in str(e["loc"]) for e in errors)


class TestInvestDebateStateValidation:

    def test_default_state_valid(self) -> None:
        state = InvestDebateState()
        assert state.bull_history == []
        assert state.bear_history == []
        assert state.current_round == 0
        assert state.max_rounds == 3
        assert state.investment_plan is None

    def test_populated_state_valid(self) -> None:
        state = InvestDebateState(
            bull_history=["Strong earnings growth", "Expanding market share"],
            bear_history=["High valuation", "Sector headwinds"],
            current_round=2,
            max_rounds=3,
            investment_plan="Cautious buy with tight stop-loss",
        )
        assert len(state.bull_history) == 2
        assert len(state.bear_history) == 2
        assert state.current_round == 2

    def test_rejects_invalid_round_type(self) -> None:
        with pytest.raises(ValidationError):
            InvestDebateState(current_round="not-a-number")  # type: ignore[arg-type]


class TestRiskDebateStateValidation:

    def test_default_state_valid(self) -> None:
        state = RiskDebateState()
        assert state.aggressive_assessment is None
        assert state.neutral_assessment is None
        assert state.conservative_assessment is None
        assert state.current_round == 0
        assert state.max_rounds == 2
        assert state.risk_debate_history == []
        assert state.final_risk_decision is None

    def test_populated_state_valid(self) -> None:
        state = RiskDebateState(
            aggressive_assessment="Risk acceptable, position 8% of portfolio",
            neutral_assessment="Risk moderate, position 5% of portfolio",
            conservative_assessment="Risk high, position 2% of portfolio",
            current_round=1,
            max_rounds=2,
            risk_debate_history=["Round 1 summary"],
            final_risk_decision="Position 5% with tight stop-loss",
        )
        assert state.aggressive_assessment is not None
        assert state.neutral_assessment is not None
        assert state.conservative_assessment is not None

    def test_rejects_invalid_max_rounds_type(self) -> None:
        with pytest.raises(ValidationError):
            RiskDebateState(max_rounds="two")  # type: ignore[arg-type]


class TestAgentStateWithDebateNesting:

    def test_agent_state_with_invest_debate(self) -> None:
        debate = InvestDebateState(
            bull_history=["Growth thesis"],
            bear_history=["Valuation concern"],
            current_round=1,
        )
        state = AgentState(
            ticker="RELIANCE",
            date="2026-04-07",
            investment_debate_state=debate,
        )
        assert state.investment_debate_state is not None
        assert state.investment_debate_state.bull_history == ["Growth thesis"]

    def test_agent_state_with_risk_debate(self) -> None:
        risk = RiskDebateState(
            aggressive_assessment="Go for it",
            current_round=1,
        )
        state = AgentState(
            ticker="RELIANCE",
            date="2026-04-07",
            risk_debate_state=risk,
        )
        assert state.risk_debate_state is not None
        assert state.risk_debate_state.aggressive_assessment == "Go for it"

    def test_agent_state_rejects_invalid_debate_type(self) -> None:
        state = AgentState(
            ticker="RELIANCE",
            investment_debate_state={"bull_history": ["test"], "current_round": 1},  # type: ignore[arg-type]
        )
        assert state.investment_debate_state is not None
        assert state.investment_debate_state.current_round == 1
