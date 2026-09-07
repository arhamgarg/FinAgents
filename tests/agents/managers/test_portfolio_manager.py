from __future__ import annotations

from langchain_core.runnables import RunnableConfig

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.analysts.schemas import TechnicalReport
from finagents.agents.managers.schemas import PortfolioContext, PortfolioDecision
from finagents.agents.trader.schemas import TradingSignal
from finagents.agents.utils.agent_states import AgentState, RiskDebateState
from finagents.config import Settings




@pytest.fixture
def default_settings() -> Settings:
    return Settings(llm_provider="openrouter", openrouter_api_key="fake")


@pytest.fixture
def trader_signal_buy() -> TradingSignal:
    return TradingSignal(
        action="BUY",
        conviction=0.7,
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=120.0,
        position_size_pct=0.03,
        position_size_shares=30,
        rationale="Strong breakout on volume expansion.",
    )


@pytest.fixture
def technical_report_healthy() -> TechnicalReport:
    return TechnicalReport(
        current_price=100.0,
        confidence=0.7,
        circuit_band_pct=0.20,
        t2t_flag=False,
        atr_14=2.5,
    )


@pytest.fixture
def state_with_signal(
    trader_signal_buy: TradingSignal,
    technical_report_healthy: TechnicalReport,
) -> AgentState:
    return AgentState(
        ticker="TEST",
        date="2026-04-21",
        trader_signal=trader_signal_buy,
        technical_report=technical_report_healthy,
        risk_debate_state=RiskDebateState(
            risk_debate_history={0: "consolidation text"},
            current_round=1,
            max_rounds=1,
            synthesized_position_size_pct=0.025,
        ),
        portfolio_context=PortfolioContext(
            holdings={"TEST": 100},
            sector_exposure={"IT": 0.40},
            total_value_inr=500_000.0,
        ),
    )


def _build_mock_llm(decision: PortfolioDecision) -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=decision)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _build_failing_mock_llm(*side_effect_errors: Exception) -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=list(side_effect_errors))
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm




class TestPortfolioManagerNode:

    def test_writes_final_trade_decision(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(approve_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert "final_trade_decision" in result

    def test_uses_deep_think_tier(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )
        mock_llm = _build_mock_llm(approve_decision)
        captured = {}

        def _capture(tier, settings, *args, **kwargs):
            captured["tier"] = tier
            captured["kwargs"] = kwargs
            return mock_llm

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            side_effect=_capture,
        ):
            asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert captured["tier"] == "deep_think"

    def test_emits_portfolio_decision(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(approve_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert isinstance(result["final_trade_decision"], PortfolioDecision)

    def test_prompt_contains_portfolio_context(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import build_pm_prompt

        messages = build_pm_prompt(
            trader_signal=state_with_signal.trader_signal,
            risk_debate_state=state_with_signal.risk_debate_state,
            portfolio_context=state_with_signal.portfolio_context,
            settings=default_settings,
        )
        human_content = messages[1].content
        assert isinstance(human_content, str)
        assert "TEST" in human_content or "holdings" in human_content
        assert "500000" in human_content or "total_value_inr" in human_content

    def test_prompt_contains_trader_signal_json(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import build_pm_prompt

        messages = build_pm_prompt(
            trader_signal=state_with_signal.trader_signal,
            risk_debate_state=state_with_signal.risk_debate_state,
            portfolio_context=state_with_signal.portfolio_context,
            settings=default_settings,
        )
        human_content = messages[1].content
        assert isinstance(human_content, str)
        assert "BUY" in human_content
        assert "breakout" in human_content

    def test_prompt_contains_modify_surface_rule(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import build_pm_prompt

        messages = build_pm_prompt(
            trader_signal=state_with_signal.trader_signal,
            risk_debate_state=state_with_signal.risk_debate_state,
            portfolio_context=state_with_signal.portfolio_context,
            settings=default_settings,
        )
        system_content = messages[0].content
        assert isinstance(system_content, str)
        assert "MODIFY SURFACE RULE" in system_content
        assert "" in system_content

    def test_prompt_contains_anti_rubber_stamp_directive(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import build_pm_prompt

        messages = build_pm_prompt(
            trader_signal=state_with_signal.trader_signal,
            risk_debate_state=state_with_signal.risk_debate_state,
            portfolio_context=state_with_signal.portfolio_context,
            settings=default_settings,
        )
        system_content = messages[0].content
        assert isinstance(system_content, str)
        assert "ANTI-RUBBER-STAMP" in system_content
        assert "cite a specific" in system_content.lower() or (
            "portfolio-state factor" in system_content
        )

    def test_retry_once_on_llm_failure(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=[RuntimeError("first-fail"), approve_decision])
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=llm,
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert structured.ainvoke.call_count == 2
        assert result["final_trade_decision"].decision == "approve"

    def test_crash_to_veto_fallback(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        mock_llm = _build_failing_mock_llm(RuntimeError("first-err"), RuntimeError("second-err"))
        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        decision = result["final_trade_decision"]
        assert isinstance(decision, PortfolioDecision)
        assert decision.decision == "veto"
        assert decision.approved is False
        assert "OutputFixingParser failure" in decision.pm_notes




class TestModifySurfaceClamp:

    def test_modify_reverts_entry_price(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        tampered_signal = trader_signal_buy.model_copy()
        tampered_signal.entry_price = 150.0
        tampered_signal.position_size_pct = 0.02
        tampered_signal.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered_signal,
            approved=False,
            decision="modify",
            pm_notes="reduce size due to sector concentration",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.entry_price == 100.0

    def test_modify_reverts_stop_loss(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        tampered = trader_signal_buy.model_copy()
        tampered.stop_loss = 80.0
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="size reduction only",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.stop_loss == 95.0

    def test_modify_reverts_targets(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        tampered = trader_signal_buy.model_copy()
        tampered.target_1 = 200.0
        tampered.target_2 = 300.0
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="size reduction only",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.target_1 == 110.0
        assert final_signal.target_2 == 120.0

    def test_modify_reverts_action(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        tampered = trader_signal_buy.model_copy()
        tampered.action = "SELL"
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="attempted action flip via modify",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.action == "BUY"

    def test_modify_preserves_new_position_size(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        tampered = trader_signal_buy.model_copy()
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="reduce position due to sector concentration",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.position_size_pct == 0.02
        assert final_signal.position_size_shares == 20




class TestValidatorRerun:

    def test_circuit_band_breach_flips_modify_to_veto(
        self,
        trader_signal_buy: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        narrow_band_state = AgentState(
            ticker="TEST",
            date="2026-04-21",
            trader_signal=trader_signal_buy,
            technical_report=TechnicalReport(
                current_price=100.0,
                confidence=0.7,
                circuit_band_pct=0.05,
                t2t_flag=False,
                atr_14=2.5,
            ),
            risk_debate_state=RiskDebateState(
                risk_debate_history={0: "consolidation"},
                current_round=1,
                max_rounds=1,
                synthesized_position_size_pct=0.025,
            ),
            portfolio_context=PortfolioContext(total_value_inr=500_000.0),
        )

        tampered = trader_signal_buy.model_copy()
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="reduce size",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(narrow_band_state, config))

        decision = result["final_trade_decision"]
        assert decision.decision == "veto"
        assert decision.approved is False
        assert "[PM modify rejected by validator]" in decision.pm_notes

    def test_stt_floor_breach_flips_modify_to_veto(
        self,
        trader_signal_buy: TradingSignal,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        thin_target_trader = trader_signal_buy.model_copy()
        thin_target_trader.target_1 = 100.2

        state = AgentState(
            ticker="TEST",
            date="2026-04-21",
            trader_signal=thin_target_trader,
            technical_report=TechnicalReport(
                current_price=100.0,
                confidence=0.7,
                circuit_band_pct=0.20,
                t2t_flag=False,
                atr_14=2.5,
            ),
            risk_debate_state=RiskDebateState(
                risk_debate_history={0: "consolidation"},
                current_round=1,
                max_rounds=1,
                synthesized_position_size_pct=0.025,
            ),
            portfolio_context=PortfolioContext(total_value_inr=500_000.0),
        )

        tampered = thin_target_trader.model_copy()
        tampered.position_size_pct = 0.02
        tampered.position_size_shares = 20
        pm_decision = PortfolioDecision(
            signal=tampered,
            approved=False,
            decision="modify",
            pm_notes="modify size",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(pm_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state, config))

        decision = result["final_trade_decision"]
        assert decision.decision == "veto"
        assert decision.approved is False
        assert "[PM modify rejected by validator]" in decision.pm_notes

    def test_approve_path_does_not_rerun_validators(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with (
            patch(
                "finagents.agents.managers.portfolio_manager.create_llm_client",
                return_value=_build_mock_llm(approve_decision),
            ),
            patch(
                "finagents.agents.managers.portfolio_manager.run_validators"
            ) as mock_run_validators,
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert mock_run_validators.call_count == 0
        assert result["final_trade_decision"].decision == "approve"




class TestVetoShape:

    def test_veto_preserves_entry_stop_targets_rationale(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        junk_signal = TradingSignal(
            action="SELL",
            conviction=0.0,
            entry_price=999.0,
            stop_loss=888.0,
            target_1=777.0,
            rationale="veto rationale",
        )
        veto_decision = PortfolioDecision(
            signal=junk_signal,
            approved=False,
            decision="veto",
            pm_notes="sector overweight at 40%",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(veto_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.entry_price == trader_signal_buy.entry_price == 100.0
        assert final_signal.stop_loss == trader_signal_buy.stop_loss == 95.0
        assert final_signal.target_1 == trader_signal_buy.target_1 == 110.0
        assert final_signal.target_2 == trader_signal_buy.target_2 == 120.0
        assert final_signal.rationale == trader_signal_buy.rationale

    def test_veto_zeros_position_size_pct(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        veto_signal = trader_signal_buy.model_copy()
        veto_signal.position_size_pct = 0.10
        veto_signal.position_size_shares = 100
        veto_decision = PortfolioDecision(
            signal=veto_signal,
            approved=False,
            decision="veto",
            pm_notes="sector overweight",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(veto_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.position_size_pct is None

    def test_veto_zeros_position_size_shares(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        veto_signal = trader_signal_buy.model_copy()
        veto_signal.position_size_shares = 100
        veto_decision = PortfolioDecision(
            signal=veto_signal,
            approved=False,
            decision="veto",
            pm_notes="sector overweight",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(veto_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.position_size_shares is None

    def test_veto_sets_action_hold(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        veto_signal = trader_signal_buy.model_copy()
        veto_signal.action = "BUY"
        veto_decision = PortfolioDecision(
            signal=veto_signal,
            approved=False,
            decision="veto",
            pm_notes="sector overweight",
        )

        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(veto_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        final_signal = result["final_trade_decision"].signal
        assert final_signal.action == "HOLD"




class TestReturnType:

    def test_returns_portfolio_decision_not_dict(
        self,
        state_with_signal: AgentState,
        default_settings: Settings,
        trader_signal_buy: TradingSignal,
    ) -> None:
        from finagents.agents.managers.portfolio_manager import (
            portfolio_manager_node,
        )

        approve_decision = PortfolioDecision(
            signal=trader_signal_buy.model_copy(),
            approved=True,
            decision="approve",
            pm_notes="low concentration in IT sector",
        )
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.managers.portfolio_manager.create_llm_client",
            return_value=_build_mock_llm(approve_decision),
        ):
            result = asyncio.run(portfolio_manager_node(state_with_signal, config))

        assert isinstance(result["final_trade_decision"], PortfolioDecision)
        assert not isinstance(result["final_trade_decision"], dict)
