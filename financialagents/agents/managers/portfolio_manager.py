from __future__ import annotations

from typing import Any, Optional, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.managers.schemas import PortfolioContext, PortfolioDecision
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.agents.trader.validators import (
    ValidatorContext,
    run_validators,
)
from financialagents.agents.utils.agent_states import AgentState, RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


def build_pm_prompt(
    trader_signal: Optional[TradingSignal],
    risk_debate_state: Optional[RiskDebateState],
    portfolio_context: Optional[PortfolioContext],
    settings: Settings,
) -> list[BaseMessage]:
    last_consolidation = ""
    synthesized: Optional[float] = None
    if risk_debate_state is not None:
        turns = risk_debate_state.risk_turns()
        if turns:
            last_consolidation = turns[-1]
        synthesized = risk_debate_state.synthesized_position_size_pct

    portfolio_json = (
        portfolio_context.model_dump_json(indent=2) if portfolio_context
        else '{"holdings":{},"sector_exposure":{},"total_value_inr":0.0}'
    )
    trader_json = (
        trader_signal.model_dump_json(indent=2) if trader_signal
        else '(no trader signal)'
    )

    system_content = (
        "You are the Portfolio Manager for an Indian equity pipeline. You read "
        "(1) the trader's TradingSignal, (2) the risk debate consolidation "
        "(textual synthesis + deterministic synthesized_position_size_pct), and "
        "(3) the portfolio context (holdings, sector exposure, total value). "
        "Emit a PortfolioDecision:\n"
        " - decision in {approve, modify, veto}\n"
        " - approved: bool (True only if decision=='approve')\n"
        " - signal: final TradingSignal\n"
        " - pm_notes: non-generic reasoning that cites a portfolio-state "
        " factor (e.g., 'low concentration in this sector', 'cash headroom "
        " available', 'sector overweight at X%').\n\n"
        "MODIFY SURFACE RULE modify can change ONLY position_size_pct "
        "and position_size_shares. To change action, entry_price, stop_loss, "
        "or targets, use decision='veto'.\n\n"
        "ANTI-RUBBER-STAMP on approve, pm_notes MUST cite a specific "
        "portfolio-state factor. A generic 'looks good' is insufficient."
    )
    human_content = (
        "## Trader Signal\n\n"
        f"{trader_json}\n\n"
        "## Risk Debate Consolidation\n\n"
        f"{last_consolidation or '(no consolidation yet)'}\n\n"
        f"Synthesized position size (from risk personas, Settings-weighted): "
        f"{synthesized if synthesized is not None else '(none)'}\n\n"
        "## Portfolio Context\n\n"
        f"{portfolio_json}\n\n"
        "## Your Task\n\nEmit a PortfolioDecision."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


def _clamp_modify_surface(
    decision: PortfolioDecision, original: Optional[TradingSignal]
) -> PortfolioDecision:
    if original is None:
        return decision
    modified = decision.signal.model_copy()
    modified.action = original.action
    modified.entry_price = original.entry_price
    modified.stop_loss = original.stop_loss
    modified.target_1 = original.target_1
    modified.target_2 = original.target_2
    modified.rationale = original.rationale
    modified.circuit_band_note = original.circuit_band_note
    decision.signal = modified
    return decision


def _apply_veto_shape(
    signal: TradingSignal, original: Optional[TradingSignal]
) -> TradingSignal:
    if original is None:
        signal.action = "HOLD"
        signal.position_size_pct = None
        signal.position_size_shares = None
        return signal
    vetoed = original.model_copy()
    vetoed.action = "HOLD"
    vetoed.position_size_pct = None
    vetoed.position_size_shares = None
    return vetoed


def _veto_fallback(
    trader_signal: Optional[TradingSignal], pm_notes: str
) -> PortfolioDecision:
    if trader_signal is None:
        signal = TradingSignal(
            action="HOLD",
            conviction=0.0,
            rationale="No trader signal available",
        )
    else:
        signal = trader_signal.model_copy()
        signal.action = "HOLD"
        signal.position_size_pct = None
        signal.position_size_shares = None
    return PortfolioDecision(
        signal=signal,
        approved=False,
        decision="veto",
        pm_notes=pm_notes,
    )


async def portfolio_manager_node(
    state: AgentState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_pm_prompt(
        trader_signal=state.trader_signal,
        risk_debate_state=state.risk_debate_state,
        portfolio_context=state.portfolio_context,
        settings=settings,
    )

    llm = create_llm_client(
        "deep_think",
        settings,
        call_site="portfolio_manager",
    )
    structured_llm = llm.with_structured_output(PortfolioDecision)

    decision: PortfolioDecision
    try:
        decision = cast(PortfolioDecision, await structured_llm.ainvoke(prompt))
    except Exception as first_err:
        try:
            decision = cast(PortfolioDecision, await structured_llm.ainvoke(prompt))
        except Exception:
            decision = _veto_fallback(
                state.trader_signal,
                pm_notes=f"OutputFixingParser failure: {first_err}",
            )
            return {"final_trade_decision": decision}

    if decision.decision == "modify":
        decision = _clamp_modify_surface(decision, original=state.trader_signal)

        current_price = (
            state.technical_report.current_price if state.technical_report else None
        )
        circuit_band_pct = (
            state.technical_report.circuit_band_pct if state.technical_report else None
        )
        ctx = ValidatorContext(
            current_price=current_price,
            circuit_band_pct=circuit_band_pct,
            settings=settings,
        )
        validated_signal = run_validators(decision.signal, ctx)
        if validated_signal.action == "HOLD":
            guard_note = getattr(validated_signal, "circuit_band_note", None) or ""
            rationale_tail = (validated_signal.rationale or "")[:200]
            original_pm_notes = decision.pm_notes
            decision.decision = "veto"
            decision.approved = False
            decision.signal = validated_signal
            decision.pm_notes = (
                f"[PM modify rejected by validator] "
                f"{guard_note} {rationale_tail} | PM original note: {original_pm_notes}"
            ).strip()
        else:
            decision.signal = validated_signal

    if decision.decision == "veto":
        decision.signal = _apply_veto_shape(decision.signal, original=state.trader_signal)
        decision.approved = False

    return {"final_trade_decision": decision}
