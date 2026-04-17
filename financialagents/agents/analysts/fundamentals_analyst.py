from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.analysts.schemas import FundamentalsAnalystReport
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.dataflows.fundamentals_schema import FundamentalsReport
from financialagents.dataflows.interface import get_fundamentals
from financialagents.llm_clients.factory import create_llm_client


def build_fundamentals_prompt(
    ticker: str, report: FundamentalsReport
) -> list[BaseMessage]:
    system_content = (
        "You are an expert Indian equity fundamental analyst with deep knowledge of "
        "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) listed companies. "
        "You follow SEBI regulations and Indian market conventions.\n\n"
        "Indian valuation benchmarks (as of 2024):\n"
        "- Nifty 50 average P/E: ~22x (P/E >30 is expensive; P/E <15 may indicate value)\n"
        "- P/B ratio: most quality Indian large-caps trade at 2-5x; <1x may signal distress\n"
        "- ROCE >15% is considered healthy for Indian industrials and IT companies\n"
        "- Debt-to-Equity (D/E) <1 is preferred; D/E >2 signals high leverage risk\n"
        "- Promoter holding >50% generally indicates management confidence\n"
        "- Promoter pledge >20% is a red flag\n"
        "- Dividend yield >2% is above average for Indian equities\n\n"
        "SEBI norms: Quarterly results are mandatory for listed companies. "
        "Insider trading regulations require disclosure of promoter transactions. "
        "Substantial acquisition (>5%) requires public disclosure.\n\n"
        "You MUST produce a FundamentalsAnalystReport JSON with ALL fields populated. "
        "For numeric fields, use the exact values from the data provided below. "
        "For assessment fields, provide concise Indian-market-aware analysis. "
        "Confidence should reflect data quality: 1.0 = full data, 0.5 = partial, 0.0 = no data. "
        "Set data_sources after you receive the prompt — it will be populated by the node."
    )

    v = report.valuation
    p = report.profitability
    d = report.debt_balance
    s = report.shareholding

    def fmt_opt(val: float | None, suffix: str = "") -> str:
        return f"{val:.2f}{suffix}" if val is not None else "N/A"

    quarterly_str = ""
    if report.quarterly_trend:
        rows = []
        for q in report.quarterly_trend:
            rev = fmt_opt(q.revenue_cr, " Cr")
            pat = fmt_opt(q.pat_cr, " Cr")
            rows.append(f"  {q.quarter}: Revenue={rev}, PAT={pat}")
        quarterly_str = "\n".join(rows)
    else:
        quarterly_str = " No quarterly data available"

    human_content = (
        f"## Fundamentals Analysis Request: {ticker}\n\n"
        f"### Valuation Metrics\n"
        f"  P/E Ratio: {fmt_opt(v.pe_ratio)}x\n"
        f"  P/B Ratio: {fmt_opt(v.pb_ratio)}x\n"
        f"  EV/EBITDA: {fmt_opt(v.ev_ebitda)}x\n"
        f"  Market Cap: ₹{fmt_opt(v.market_cap_cr)} Cr\n"
        f"  Dividend Yield: {fmt_opt(v.dividend_yield, '%')}\n\n"
        f"### Profitability Metrics\n"
        f"  ROCE: {fmt_opt(p.roce, '%')}\n"
        f"  ROE: {fmt_opt(p.roe, '%')}\n"
        f"  Operating Margin: {fmt_opt(p.operating_margin, '%')}\n"
        f"  Net Margin: {fmt_opt(p.net_margin, '%')}\n\n"
        f"### Debt & Balance Sheet\n"
        f"  Debt-to-Equity: {fmt_opt(d.debt_to_equity)}x\n"
        f"  Interest Coverage: {fmt_opt(d.interest_coverage)}x\n"
        f"  Current Ratio: {fmt_opt(d.current_ratio)}x\n\n"
        f"### Shareholding Pattern\n"
        f"  Promoter Holding: {fmt_opt(s.promoter_holding_pct, '%')}\n"
        f"  Promoter Pledge: {fmt_opt(s.promoter_pledge_pct, '%')}\n"
        f"  FII Holding: {fmt_opt(s.fii_holding_pct, '%')}\n"
        f"  DII Holding: {fmt_opt(s.dii_holding_pct, '%')}\n\n"
        f"### Quarterly Trend (Last 4 Quarters)\n"
        f"{quarterly_str}\n\n"
        f"### Instructions\n\n"
        f"Based on the above data, produce a complete FundamentalsAnalystReport JSON with:\n"
        f"- pe_ratio, pb_ratio, roce, debt_to_equity, promoter_holding_pct: "
        f"copy exact values from above\n"
        f"- quarterly_result_summary: 1-2 sentence summary of recent quarterly trends\n"
        f"- fundamental_assessment: 3-4 sentence Indian-market-aware assessment of "
        f"{ticker}'s fundamental health (compare vs Nifty 50 benchmarks)\n"
        f"- confidence: float 0.0-1.0 based on data completeness\n"
        f"- data_sources: leave as empty list (populated by the node)\n"
        f"- notes: any caveats, data limitations, or flags (e.g., high promoter pledge)"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]


async def fundamentals_analyst_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    result = await get_fundamentals(state.ticker, settings=settings)

    if result.status in ("error", "empty") or result.data is None:
        return {
            "fundamentals_report": FundamentalsAnalystReport(
                confidence=0.0,
                data_sources=[],
                notes=f"No fundamentals data: {result.error or 'empty'}",
            )
        }

    prompt = build_fundamentals_prompt(state.ticker, result.data)

    llm = create_llm_client("quick_think", settings)
    structured_llm = llm.with_structured_output(FundamentalsAnalystReport)

    report: FundamentalsAnalystReport
    try:
        report = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            report = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            raise retry_err from first_err

    if result.source and result.source not in report.data_sources:
        report.data_sources = list(report.data_sources) + [result.source]

    return {"fundamentals_report": report}
