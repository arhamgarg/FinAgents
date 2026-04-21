from __future__ import annotations

import pytest
from pydantic import ValidationError

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    TechnicalReport,
    NewsReport,
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
        from financialagents.agents.managers.schemas import PortfolioDecision
        from financialagents.agents.trader.schemas import TradingSignal

        state = AgentState(
            ticker="TCS",
            date="2026-04-07",
            sector="IT",
            selected_tickers=["TCS", "INFY", "WIPRO"],
            technical_report=TechnicalReport(rsi=65.0, macd_signal="bullish_crossover", confidence=0.8),
            news_report=NewsReport(article_count=3, sentiment_direction="positive", confidence=0.7),
            fundamentals_report=FundamentalsAnalystReport(pe_ratio=28.0, roce=45.0, confidence=0.9),
            research_summary=ResearchBrief(consensus_view="Overall bullish outlook"),
            trader_investment_plan="Buy at 3800, SL 3700",
            final_trade_decision=PortfolioDecision(
                signal=TradingSignal(action="BUY", conviction=0.85),
                approved=True,
                decision="approve",
            ),
        )
        assert state.sector == "IT"
        assert len(state.selected_tickers) == 3
        assert state.final_trade_decision is not None
        assert state.final_trade_decision.signal.action == "BUY"
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
        assert state.bull_history == {}
        assert state.bear_history == {}
        assert state.current_round == 0
        assert state.max_rounds == 3
        assert state.investment_plan is None

    def test_populated_state_valid(self) -> None:
        state = InvestDebateState(
            bull_history={1: "Strong earnings growth", 2: "Expanding market share"},
            bear_history={1: "High valuation", 2: "Sector headwinds"},
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
        assert state.risk_debate_history == {}
        assert state.final_risk_decision is None

    def test_populated_state_valid(self) -> None:
        state = RiskDebateState(
            aggressive_assessment="Risk acceptable, position 8% of portfolio",
            neutral_assessment="Risk moderate, position 5% of portfolio",
            conservative_assessment="Risk high, position 2% of portfolio",
            current_round=1,
            max_rounds=2,
            risk_debate_history={0: "Round 1 summary"},
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
            bull_history={1: "Growth thesis"},
            bear_history={1: "Valuation concern"},
            current_round=1,
        )
        state = AgentState(
            ticker="RELIANCE",
            date="2026-04-07",
            investment_debate_state=debate,
        )
        assert state.investment_debate_state is not None
        assert state.investment_debate_state.bull_history == {1: "Growth thesis"}

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
            investment_debate_state={"bull_history": {1: "test"}, "current_round": 1},  # type: ignore[arg-type]
        )
        assert state.investment_debate_state is not None
        assert state.investment_debate_state.current_round == 1


class TestPhase4InvestDebateStateFields:

    def test_default_bull_history_is_empty_dict(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState()
        assert state.bull_history == {}
        assert isinstance(state.bull_history, dict)

    def test_default_bear_history_is_empty_dict(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState()
        assert state.bear_history == {}
        assert isinstance(state.bear_history, dict)

    def test_research_brief_snapshot_default_empty_string(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState()
        assert state.research_brief_snapshot == ""

    def test_bull_turns_accessor_orders_by_key(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState(bull_history={2: "second", 1: "first", 3: "third"})
        assert state.bull_turns() == ["first", "second", "third"]

    def test_bear_turns_accessor_orders_by_key(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState(bear_history={2: "second", 1: "first"})
        assert state.bear_turns() == ["first", "second"]

    def test_empty_accessors_return_empty_list(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState()
        assert state.bull_turns() == []
        assert state.bear_turns() == []

    def test_construction_with_dict_histories(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        state = InvestDebateState(
            bull_history={1: "bull-1"},
            bear_history={1: "bear-1"},
            current_round=1,
            max_rounds=3,
            research_brief_snapshot="brief text",
        )
        assert state.bull_history == {1: "bull-1"}
        assert state.research_brief_snapshot == "brief text"


class TestPhase4RiskDebateStateFields:

    def test_default_trader_signal_snapshot_empty_string(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        state = RiskDebateState()
        assert state.trader_signal_snapshot == ""

    def test_round_trips_via_model_validate(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        payload = {"trader_signal_snapshot": "action=BUY, conviction=0.7"}
        state = RiskDebateState.model_validate(payload)
        assert state.trader_signal_snapshot == "action=BUY, conviction=0.7"


class TestPhase4AgentStateFields:

    def test_defaults_are_none(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        state = AgentState()
        assert state.trader_signal is None
        assert state.portfolio_context is None

    def test_trader_signal_populated(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.agents.trader.schemas import TradingSignal
        sig = TradingSignal(action="BUY", conviction=0.7)
        state = AgentState(ticker="RELIANCE", date="2024-01-15", trader_signal=sig)
        assert state.trader_signal is not None
        assert state.trader_signal.action == "BUY"
        assert state.trader_signal.conviction == 0.7

    def test_portfolio_context_populated(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.agents.managers.schemas import PortfolioContext
        ctx = PortfolioContext(holdings={"TCS": 5}, total_value_inr=50000.0)
        state = AgentState(portfolio_context=ctx)
        assert state.portfolio_context is not None
        assert state.portfolio_context.holdings == {"TCS": 5}

    def test_model_validate_rehydrates_nested_trading_signal(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        payload = {
            "ticker": "TCS",
            "date": "2024-01-15",
            "trader_signal": {"action": "SELL", "conviction": 0.3},
        }
        state = AgentState.model_validate(payload)
        assert state.trader_signal is not None
        assert state.trader_signal.action == "SELL"
        assert state.trader_signal.conviction == 0.3

    def test_existing_trader_investment_plan_preserved(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        state = AgentState(trader_investment_plan="draft plan text")
        assert state.trader_investment_plan == "draft plan text"


class TestPhase42RiskDebateStateFields:

    def test_default_risk_debate_history_is_empty_dict(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        state = RiskDebateState()
        assert state.risk_debate_history == {}
        assert isinstance(state.risk_debate_history, dict)

    def test_risk_turns_accessor_orders_by_key(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        state = RiskDebateState(risk_debate_history={2: "second", 0: "first", 1: "middle"})
        assert state.risk_turns() == ["first", "middle", "second"]

    def test_risk_turns_empty_when_history_empty(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        assert RiskDebateState().risk_turns() == []

    def test_construction_with_dict_history(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        state = RiskDebateState(risk_debate_history={0: "turn-0"})
        assert state.risk_debate_history == {0: "turn-0"}

    def test_list_history_rejected_post_migration(self) -> None:
        import pydantic
        from financialagents.agents.utils.agent_states import RiskDebateState
        with pytest.raises(pydantic.ValidationError):
            RiskDebateState(risk_debate_history=["old-list-form"])  # ty: ignore[invalid-argument-type]

    def test_synthesized_position_size_pct_default_none(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        assert RiskDebateState().synthesized_position_size_pct is None

    def test_synthesized_position_size_pct_accepts_float(self) -> None:
        from financialagents.agents.utils.agent_states import RiskDebateState
        state = RiskDebateState(synthesized_position_size_pct=0.0325)
        assert state.synthesized_position_size_pct == pytest.approx(0.0325)


class TestPhase42AgentStateFinalTradeDecisionRetype:

    def test_final_trade_decision_is_portfolio_decision_type(self) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        from financialagents.agents.trader.schemas import TradingSignal
        from financialagents.agents.utils.agent_states import AgentState

        pd = PortfolioDecision(
            signal=TradingSignal(action="BUY", conviction=0.7),
            approved=True,
            decision="approve",
            pm_notes="concentration ok",
        )
        state = AgentState(final_trade_decision=pd)
        assert isinstance(state.final_trade_decision, PortfolioDecision)
        assert state.final_trade_decision.decision == "approve"

    def test_final_trade_decision_rejects_raw_dict_literal(self) -> None:
        import pydantic
        from financialagents.agents.utils.agent_states import AgentState
        with pytest.raises(pydantic.ValidationError):
            AgentState(final_trade_decision={"signal": "BUY"})  # ty: ignore[invalid-argument-type]
