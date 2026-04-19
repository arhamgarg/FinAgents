from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pandas_ta  # noqa: F401 — registers df.ta accessor extension
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.analysts.schemas import TechnicalReport
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.interface import get_ohlcv
from financialagents.llm_clients.factory import create_llm_client


def _safe_float(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    if len(series) == 0:
        return None
    val = series.iloc[-1]
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _count_non_none(values: list[object]) -> int:
    return sum(1 for v in values if v is not None)


def compute_technical_indicators(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty:
        return {k: None for k in [
            "rsi", "macd_line", "macd_signal_val", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid",
            "sma_50", "sma_200", "ema_20", "current_price",
            "atr_14",
        ]}

    rsi = df.ta.rsi(length=14)

    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
    macd_line_val = None
    macd_signal_val = None
    macd_hist_val = None
    if macd_df is not None and not macd_df.empty:
        cols = list(macd_df.columns)
        macd_col = next((c for c in cols if c.startswith("MACD_")), None)
        macdh_col = next((c for c in cols if c.startswith("MACDh_")), None)
        macds_col = next((c for c in cols if c.startswith("MACDs_")), None)
        if macd_col:
            macd_line_val = _safe_float(macd_df[macd_col])
        if macdh_col:
            macd_hist_val = _safe_float(macd_df[macdh_col])
        if macds_col:
            macd_signal_val = _safe_float(macd_df[macds_col])

    bb_df = df.ta.bbands(length=20, std=2)
    bb_upper_val = None
    bb_lower_val = None
    bb_mid_val = None
    if bb_df is not None and not bb_df.empty:
        cols = list(bb_df.columns)
        bbu_col = next((c for c in cols if c.startswith("BBU_")), None)
        bbl_col = next((c for c in cols if c.startswith("BBL_")), None)
        bbm_col = next((c for c in cols if c.startswith("BBM_")), None)
        if bbu_col:
            bb_upper_val = _safe_float(bb_df[bbu_col])
        if bbl_col:
            bb_lower_val = _safe_float(bb_df[bbl_col])
        if bbm_col:
            bb_mid_val = _safe_float(bb_df[bbm_col])

    sma_50 = df.ta.sma(length=50)
    sma_200 = df.ta.sma(length=200)
    ema_20 = df.ta.ema(length=20)

    atr_series = df.ta.atr(length=14)

    return {
        "rsi": _safe_float(rsi),
        "macd_line": macd_line_val,
        "macd_signal_val": macd_signal_val,
        "macd_hist": macd_hist_val,
        "bb_upper": bb_upper_val,
        "bb_lower": bb_lower_val,
        "bb_mid": bb_mid_val,
        "sma_50": _safe_float(sma_50),
        "sma_200": _safe_float(sma_200),
        "ema_20": _safe_float(ema_20),
        "current_price": _safe_float(df["close"]),
        "atr_14": _safe_float(atr_series),
    }


def build_technical_analyst_prompt(
    ticker: str,
    indicators: dict[str, float | None],
    data_result: DataResult,  # type: ignore[type-arg]
) -> list[BaseMessage]:
    system_content = (
        "You are an expert Indian equity technical analyst with deep knowledge of "
        "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) listed equities. "
        "You follow SEBI regulations and Indian market conventions.\n\n"
        "Market hours: 9:15 AM - 3:30 PM IST (Monday-Friday). "
        "The primary index benchmarks are NIFTY 50 (NSE) and SENSEX (BSE).\n\n"
        "Interpret technical indicators using Indian market context:\n"
        "- RSI >70: overbought (potential reversal in trending Indian mid/large-caps)\n"
        "- RSI <30: oversold (potential bounce, especially in quality stocks)\n"
        "- MACD crossover: bullish when MACD line crosses above signal line\n"
        "- MACD bearish crossover: MACD line crosses below signal line\n"
        "- Bollinger Band position: near_upper suggests extended rally; near_lower suggests "
        "oversold; middle_band indicates consolidation\n"
        "- Golden cross (SMA50 > SMA200): medium-term bullish signal\n"
        "- Death cross (SMA50 < SMA200): medium-term bearish signal\n\n"
        "You MUST produce a TechnicalReport JSON object with ALL fields populated. "
        "Use the indicator values provided to fill metric fields. "
        "For text fields (rsi_interpretation, macd_signal, bollinger_position, "
        "ma_crossover, trend_assessment), provide concise Indian-market-aware analysis. "
        "Support and resistance levels should be identified from the price action context.\n\n"
        "Confidence should reflect data quality: 1.0 = full data, 0.5 = partial, 0.0 = no data.\n"
        "Set data_sources to include the OHLCV provider name."
    )

    def fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A (insufficient data)"

    current = indicators.get("current_price")
    current_str = f"{current:.2f}" if current is not None else "N/A"

    rsi_val = indicators.get("rsi")
    rsi_str = fmt(rsi_val)
    if rsi_val is not None:
        if rsi_val > 70:
            rsi_zone = "OVERBOUGHT"
        elif rsi_val < 30:
            rsi_zone = "OVERSOLD"
        else:
            rsi_zone = "NEUTRAL"
        rsi_str = f"{rsi_val:.2f} [{rsi_zone}]"

    sma_50 = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")
    ma_crossover_hint = "N/A"
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            ma_crossover_hint = "golden_cross (SMA50 above SMA200 — bullish)"
        elif sma_50 < sma_200:
            ma_crossover_hint = "death_cross (SMA50 below SMA200 — bearish)"
        else:
            ma_crossover_hint = "none (SMAs equal)"

    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_mid = indicators.get("bb_mid")
    bb_position_hint = "N/A"
    if bb_upper is not None and bb_lower is not None and current is not None:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            position_pct = (current - bb_lower) / bb_range
            if position_pct > 0.8:
                bb_position_hint = "near_upper (extended — potential reversal)"
            elif position_pct < 0.2:
                bb_position_hint = "near_lower (oversold — potential bounce)"
            else:
                bb_position_hint = "middle_band (consolidation)"

    macd_line = indicators.get("macd_line")
    macd_signal = indicators.get("macd_signal_val")
    macd_hist = indicators.get("macd_hist")
    macd_signal_hint = "N/A"
    if macd_line is not None and macd_signal is not None and macd_hist is not None:
        if macd_hist > 0:
            macd_signal_hint = "bullish_crossover (MACD above signal, histogram positive)"
        elif macd_hist < 0:
            macd_signal_hint = "bearish_crossover (MACD below signal, histogram negative)"
        else:
            macd_signal_hint = "neutral (MACD at signal line)"

    stale_note = " [DATA STALE — treat with lower confidence]" if data_result.stale else ""
    source = data_result.source or "unknown"

    human_content = (
        f"## Technical Analysis Request: {ticker}\n\n"
        f"**Data Source:** {source}{stale_note}\n"
        f"**Data Points:** {len(data_result.data) if data_result.data is not None else 0} trading days\n\n"
        f"### Computed Indicator Values\n\n"
        f"**Price:**\n"
        f"  Current Price: ₹{current_str}\n\n"
        f"**RSI (14-period):**\n"
        f"  Value: {rsi_str}\n\n"
        f"**MACD (12, 26, 9):**\n"
        f"  MACD Line: {fmt(macd_line)}\n"
        f"  Signal Line: {fmt(macd_signal)}\n"
        f"  Histogram: {fmt(macd_hist)}\n"
        f"  Signal: {macd_signal_hint}\n\n"
        f"**Bollinger Bands (20, 2σ):**\n"
        f"  Upper Band: {fmt(bb_upper)}\n"
        f"  Middle Band: {fmt(bb_mid)}\n"
        f"  Lower Band: {fmt(bb_lower)}\n"
        f"  Position: {bb_position_hint}\n\n"
        f"**Moving Averages:**\n"
        f"  SMA-50: {fmt(sma_50)}\n"
        f"  SMA-200: {fmt(sma_200)}\n"
        f"  EMA-20: {fmt(indicators.get('ema_20'))}\n"
        f"  MA Crossover: {ma_crossover_hint}\n\n"
        f"### Instructions\n\n"
        f"Based on the above indicator values, produce a complete TechnicalReport JSON with:\n"
        f"- All numeric fields filled from the values above\n"
        f"- rsi_interpretation: brief RSI reading (overbought/oversold/neutral)\n"
        f"- macd_signal: one of 'bullish_crossover', 'bearish_crossover', 'neutral'\n"
        f"- bollinger_position: one of 'near_upper', 'near_lower', 'middle_band'\n"
        f"- ma_crossover: one of 'golden_cross', 'death_cross', 'none'\n"
        f"- trend_assessment: 2-3 sentence synthesis of the overall trend for {ticker}\n"
        f"- support_levels: 2-3 key price levels below current price\n"
        f"- resistance_levels: 2-3 key price levels above current price\n"
        f"- confidence: float 0.0-1.0 based on data completeness\n"
        f"- data_sources: list containing '{source}'\n"
        f"- notes: any caveats, staleness warnings, or indicator limitations"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]


async def technical_analyst_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    end_date: datetime
    if state.date:
        try:
            end_date = datetime.strptime(state.date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            end_date = datetime.now(timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    start_date = end_date - timedelta(days=365)

    result = await get_ohlcv(state.ticker, start_date, end_date, settings=settings)

    if result.status in ("error", "empty") or result.data is None:
        raise ValueError(
            f"Technical analyst requires OHLCV data for {state.ticker}: "
            f"{result.error or 'no data available'}"
        )

    indicators = compute_technical_indicators(result.data)

    prompt = build_technical_analyst_prompt(state.ticker, indicators, result)

    llm = create_llm_client("quick_think", settings)
    structured_llm = llm.with_structured_output(TechnicalReport)

    report: TechnicalReport
    try:
        report = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            report = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            raise retry_err from first_err

    if result.source and result.source not in report.data_sources:
        report.data_sources = list(report.data_sources) + [result.source]

    report.atr_14 = indicators.get("atr_14")

    try:
        from financialagents.dataflows.surveillance_client import get_surveillance_data

        circuit_band_pct, t2t_flag = await get_surveillance_data(state.ticker)
        report.circuit_band_pct = circuit_band_pct
        report.t2t_flag = t2t_flag
    except Exception:
        pass

    report.confidence = _count_non_none([
        report.current_price,
        report.rsi,
        report.macd_line,
        report.macd_hist,
        report.bb_upper,
        report.bb_lower,
        report.sma_50,
        report.sma_200,
        report.atr_14,
    ]) / 9

    return {"technical_report": report}
