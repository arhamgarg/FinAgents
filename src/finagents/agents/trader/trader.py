from __future__ import annotations

import logging
from typing import Any, Optional, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic_core import PydanticSerializationError

from finagents.agents.trader.position_sizing import (
    atr_based_size,
    fixed_fractional_size,
    kelly_size,
)
from finagents.agents.trader.schemas import TradingSignal
from finagents.agents.trader.validators import (
    ValidatorContext,
    circuit_band_validator,
    kelly_cap_validator,
    price_grounding_validator,
    stt_cost_floor_validator,
    with_suffix,
)
from finagents.agents.utils.agent_states import AgentState
from finagents.agents.utils.llm_retry import StructuredOutputError, ainvoke_with_retry
from finagents.agents.utils.run_settings import run_settings
from finagents.config import Settings
from finagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)


def build_trader_prompt(
    research_summary: Any,
    investment_thesis: str | None,
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
            "All signals are positional delivery trades on NSE/BSE cash segment — "
            "multi-day-to-multi-week holding horizons. Do not propose intraday, scalp, or F&O "
            "approaches.\n"
            "Produce a validated TradingSignal with action (BUY/HOLD/SELL), "
            "conviction, entry_price, stop_loss, target_1, rationale, and sizing "
            "fields.\n\n"
            "DIRECTIVE: If overall_confidence >= 0.6 AND the "
            "debate (Investment Thesis) clearly favors a direction (BUY or SELL), "
            "you MUST emit that direction. Only emit HOLD when genuinely "
            "uncertain or when confidence is below 0.6. Do not default to HOLD "
            "under mild uncertainty.\n\n"
            "COST FLOOR: target_1 must yield at least 0.3% above entry_price "
            "to exceed Indian equity round-trip costs (STT + GST + brokerage + "
            "exchange + SEBI + stamp duty). Sub-0.3% targets will be forced to "
            "HOLD by a post-LLM validator.\n\n"
            f"ATTRIBUTION: The configured analyst weights are {weights}. "
            "Your TradingSignal.analyst_weight_applied field will be overwritten "
            "post-LLM with these exact values — do not invent new weights."
        )
    )

    research_text = "(no research summary available)"
    if research_summary is not None:
        try:
            research_text = research_summary.model_dump_json(indent=2)
        except PydanticSerializationError as exc:
            logger.warning(
                "ResearchBrief.model_dump_json failed; degrading to repr",
                exc_info=True,
            )
            research_text = (
                f"(serialization failed: {exc!r}); falling back to repr: {research_summary!r}"
            )

    human = HumanMessage(
        content=(
            "Investment Thesis (from bull/bear debate synthesis):\n"
            f"{investment_thesis or '(unavailable)'}\n\n"
            "Research Brief (weighted ResearchBrief — includes overall_confidence):\n"
            f"{research_text}\n\n"
            "Market Data (typed fields only):\n"
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
    atr_14 = getattr(technical_report, "atr_14", None) if technical_report else None
    tech_current_price = (
        getattr(technical_report, "current_price", None) if technical_report else None
    )
    entry_price = signal.entry_price if signal.entry_price is not None else tech_current_price
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
            signal = signal.model_copy(
                update={
                    "sizing_method": "atr_based",
                    "position_size_pct": pct,
                    "position_size_shares": shares,
                }
            )
        elif stop_loss is not None:
            pct, shares = fixed_fractional_size(
                capital=capital,
                risk_per_trade_pct=risk_pct,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
            signal = signal.model_copy(
                update={
                    "sizing_method": "fixed_fractional",
                    "position_size_pct": pct,
                    "position_size_shares": shares,
                    "rationale": with_suffix(
                        signal.rationale, " [ATR unavailable; fallback to fixed_fractional]"
                    ),
                }
            )
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
        shares = int((pct * capital) // entry_price) if entry_price > 0 else 0
        signal = signal.model_copy(
            update={
                "sizing_method": "kelly",
                "position_size_pct": pct,
                "position_size_shares": max(shares, 0),
            }
        )
    elif stop_loss is not None:
        pct, shares = fixed_fractional_size(
            capital=capital,
            risk_per_trade_pct=risk_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )
        signal = signal.model_copy(
            update={
                "sizing_method": "fixed_fractional",
                "position_size_pct": pct,
                "position_size_shares": shares,
            }
        )

    shares_unset_or_zero = signal.position_size_shares is None or signal.position_size_shares == 0
    pct_unset = signal.position_size_pct is None
    if shares_unset_or_zero or pct_unset:
        if stop_loss is None:
            reason = "sizing inputs insufficient (stop_loss missing)"
        elif method == "kelly" and (signal.target_1 is None):
            reason = "sizing inputs insufficient (kelly requires target_1)"
        else:
            reason = "sizing yielded 0 shares (capital too small or stop too wide)"
        signal = signal.model_copy(
            update={
                "action": "HOLD",
                "position_size_pct": None,
                "position_size_shares": None,
                "rationale": with_suffix(signal.rationale, f" [HOLD: {reason}]"),
            }
        )
    return signal


async def trader_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:  # noqa: UP045
    settings = run_settings(config)

    technical_report = state.technical_report
    current_price = technical_report.current_price if technical_report is not None else None
    atr_14 = technical_report.atr_14 if technical_report is not None else None
    circuit_band_pct = technical_report.circuit_band_pct if technical_report is not None else None
    t2t_flag = technical_report.t2t_flag if technical_report is not None else False
    research_summary = state.research_summary
    investment_thesis = None
    if state.investment_debate_state is not None:
        investment_thesis = getattr(state.investment_debate_state, "investment_thesis", None)

    prompt = build_trader_prompt(
        research_summary=research_summary,
        investment_thesis=investment_thesis,
        current_price=current_price,
        atr_14=atr_14,
        circuit_band_pct=circuit_band_pct,
        t2t_flag=t2t_flag,
        settings=settings,
    )

    llm = create_llm_client("deep_think", settings, call_site="trader")
    structured_llm = llm.with_structured_output(TradingSignal)

    signal: TradingSignal
    try:
        signal = await ainvoke_with_retry(
            structured_llm, prompt, lambda raw: cast(TradingSignal, raw)
        )
    except StructuredOutputError as err:
        signal = TradingSignal(
            action="HOLD",
            conviction=0.0,
            rationale=(
                f"LLM invocation failed after retry: {err.retry!r} (initial: {err.first!r})"
            ),
        )

    signal = signal.model_copy(
        update={
            "analyst_weight_applied": dict(settings.analyst_weights),
            "t2t_flag": bool(t2t_flag),
        }
    )

    ctx = ValidatorContext(
        current_price=current_price,
        circuit_band_pct=circuit_band_pct,
        settings=settings,
    )
    signal = price_grounding_validator(signal, ctx).signal

    if signal.action != "HOLD":
        signal = _apply_position_sizing(signal, technical_report, settings)

    for validator in (
        circuit_band_validator,
        stt_cost_floor_validator,
        kelly_cap_validator,
    ):
        signal = validator(signal, ctx).signal

    if (
        signal.action != "HOLD"
        and signal.entry_price is not None
        and signal.stop_loss is not None
        and signal.target_1 is not None
    ):
        denom = signal.entry_price - signal.stop_loss
        if denom != 0:
            signal = signal.model_copy(
                update={"risk_reward_ratio": (signal.target_1 - signal.entry_price) / denom}
            )

    return {
        "trader_signal": signal,
        "trader_investment_thesis": signal.rationale or None,
    }
