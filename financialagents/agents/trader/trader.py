from __future__ import annotations

from typing import Any, Optional, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.trader.position_sizing import (
    atr_based_size,
    fixed_fractional_size,
    kelly_size,
)
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.agents.trader.validators import (
    ValidatorContext,
    run_validators,
)
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


def build_trader_prompt(
    research_summary: Any,
    investment_plan: str | None,
    current_price: float | None,
    atr_14: float | None,
    circuit_band_pct: float | None,
    t2t_flag: bool,
    settings: Settings,
) -> list[BaseMessage]:
    weights = dict(settings.analyst_weights)
    system = SystemMessage(
        content=(
            "You are a disciplined discretionary equity trader for NSE/BSE equities.\n"
            "All signals are positional delivery trades on NSE/BSE cash segment — multi-day-to-multi-week holding horizons. Do not propose intraday, scalp, or F&O approaches.\n"
            "Produce a validated TradingSignal with action (BUY/HOLD/SELL), "
            "conviction, entry_price, stop_loss, target_1, rationale, and sizing "
            "fields.\n\n"
            "DIRECTIVE / P10): If overall_confidence >= 0.6 AND the "
            "debate (Investment Plan) clearly favors a direction (BUY or SELL), "
            "you MUST emit that direction. Only emit HOLD when genuinely "
            "uncertain or when confidence is below 0.6. Do not default to HOLD "
            "under mild uncertainty.\n\n"
            " / P7: target_1 must yield at least 0.3% above entry_price "
            "to exceed Indian equity round-trip costs (STT + GST + brokerage + "
            "exchange + SEBI + stamp duty). Sub-0.3% targets will be forced to "
            "HOLD by a post-LLM validator.\n\n"
            f" ATTRIBUTION: The configured analyst weights are {weights}. "
            "Your TradingSignal.analyst_weight_applied field will be overwritten "
            "post-LLM with these exact values — do not invent new weights."
        )
    )

    research_text = "(no research summary available)"
    if research_summary is not None:
        try:
            research_text = research_summary.model_dump_json(indent=2)
        except Exception:
            research_text = str(research_summary)

    human = HumanMessage(
        content=(
            "Investment Plan (from bull/bear debate synthesis):\n"
            f"{investment_plan or '(unavailable)'}\n\n"
            "Research Brief (weighted ResearchBrief — includes overall_confidence):\n"
            f"{research_text}\n\n"
            "Market Data (typed fields only — boundary):\n"
            f"  current_price: {current_price}\n"
            f"  atr_14: {atr_14}\n"
            f"  circuit_band_pct: {circuit_band_pct}\n"
            f"  t2t_flag: {t2t_flag}\n"
        )
    )
    return [system, human]


def _apply_position_sizing(
    signal: TradingSignal,
    technical_report: Any,
    settings: Settings,
) -> TradingSignal:
    capital = settings.portfolio_value_inr
    risk_pct = settings.risk_per_trade_pct
    method = settings.position_sizing_method
    atr_14 = (
        getattr(technical_report, "atr_14", None) if technical_report else None
    )
    tech_current_price = (
        getattr(technical_report, "current_price", None) if technical_report else None
    )
    entry_price = (
        signal.entry_price if signal.entry_price is not None else tech_current_price
    )
    stop_loss = signal.stop_loss

    if entry_price is None:
        return signal

    if method == "atr_based":
        if atr_14 is not None and atr_14 > 0:
            pct, shares = atr_based_size(
                capital=capital,
                risk_per_trade_pct=risk_pct,
                atr=atr_14,
                atr_multiplier=settings.atr_stop_multiplier,
                entry_price=entry_price,
            )
            signal.sizing_method = "atr_based"
            signal.position_size_pct = pct
            signal.position_size_shares = shares
        else:
            if stop_loss is not None:
                pct, shares = fixed_fractional_size(
                    capital=capital,
                    risk_per_trade_pct=risk_pct,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                )
                signal.sizing_method = "fixed_fractional"
                signal.position_size_pct = pct
                signal.position_size_shares = shares
                suffix = " [ATR unavailable; fallback to fixed_fractional]"
                if suffix not in (signal.rationale or ""):
                    signal.rationale = (signal.rationale or "") + suffix
    elif method == "kelly":
        win_rate = float(signal.conviction or 0.0)
        reward_ratio = 0.0
        if stop_loss is not None and entry_price > 0 and signal.target_1 is not None:
            denom = entry_price - stop_loss
            if denom > 0:
                reward_ratio = (signal.target_1 - entry_price) / denom
        pct = kelly_size(
            win_rate=win_rate,
            reward_ratio=reward_ratio,
            kelly_fraction=settings.kelly_fraction,
        )
        signal.sizing_method = "kelly"
        signal.position_size_pct = pct
        shares = int((pct * capital) // entry_price) if entry_price > 0 else 0
        signal.position_size_shares = max(shares, 0)
    else:
        if stop_loss is not None:
            pct, shares = fixed_fractional_size(
                capital=capital,
                risk_per_trade_pct=risk_pct,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
            signal.sizing_method = "fixed_fractional"
            signal.position_size_pct = pct
            signal.position_size_shares = shares
    return signal


async def trader_node(
    state: AgentState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    technical_report = state.technical_report
    current_price = (
        technical_report.current_price if technical_report is not None else None
    )
    atr_14 = technical_report.atr_14 if technical_report is not None else None
    circuit_band_pct = (
        technical_report.circuit_band_pct if technical_report is not None else None
    )
    t2t_flag = (
        technical_report.t2t_flag if technical_report is not None else False
    )
    research_summary = state.research_summary
    investment_plan = None
    if state.investment_debate_state is not None:
        investment_plan = getattr(
            state.investment_debate_state, "investment_plan", None
        )

    prompt = build_trader_prompt(
        research_summary=research_summary,
        investment_plan=investment_plan,
        current_price=current_price,
        atr_14=atr_14,
        circuit_band_pct=circuit_band_pct,
        t2t_flag=t2t_flag,
        settings=settings,
    )

    llm = create_llm_client("deep_think", settings)
    structured_llm = llm.with_structured_output(TradingSignal)

    signal: TradingSignal
    try:
        signal = cast(TradingSignal, await structured_llm.ainvoke(prompt))
    except Exception as first_err:
        try:
            signal = cast(TradingSignal, await structured_llm.ainvoke(prompt))
        except Exception as retry_err:
            signal = TradingSignal(
                action="HOLD",
                conviction=0.0,
                rationale=(
                    f"LLM invocation failed after retry: {retry_err!r} "
                    f"(initial: {first_err!r})"
                ),
            )

    signal.analyst_weight_applied = dict(settings.analyst_weights)

    signal.t2t_flag = bool(t2t_flag)

    if signal.action != "HOLD":
        signal = _apply_position_sizing(signal, technical_report, settings)

    ctx = ValidatorContext(
        current_price=current_price,
        circuit_band_pct=circuit_band_pct,
        settings=settings,
    )
    signal = run_validators(signal, ctx)

    if (
        signal.entry_price is not None
        and signal.stop_loss is not None
        and signal.target_1 is not None
        and signal.entry_price != signal.stop_loss
    ):
        denom = signal.entry_price - signal.stop_loss
        if denom != 0:
            signal.risk_reward_ratio = (
                signal.target_1 - signal.entry_price
            ) / denom

    return {
        "trader_signal": signal,
        "trader_investment_plan": signal.rationale or None,
    }
