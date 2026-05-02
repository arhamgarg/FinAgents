from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from financialagents.agents.managers.schemas import ResearchBrief
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clamped = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(clamped.values())
    if total == 0:
        return {k: 1.0 / len(clamped) for k in clamped}
    return {k: v / total for k, v in clamped.items()}


def build_research_manager_prompt(
    technical_report: TechnicalReport | None,
    fundamentals_report: FundamentalsAnalystReport | None,
    news_report: NewsReport | None,
    weights: dict[str, float],
) -> list[BaseMessage]:
    technical_w = weights.get("technical", 1 / 3)
    fund_w = weights.get("fundamentals", 1 / 3)
    news_w = weights.get("news", 1 / 3)

    system_content = (
        "You are the Research Manager for an Indian equity analysis pipeline. "
        "You synthesize reports from three specialist analysts into a unified investment brief. "
        "You operate in the NSE/BSE context under SEBI regulatory framework.\n\n"
        "Your role is to:\n"
        "1. Extract key findings from each analyst report\n"
        "2. Identify consensus views and conflicts across analysts\n"
        "3. Assess overall data quality and confidence\n"
        "4. Produce a ResearchBrief that serves as input for bull/bear debate\n\n"
        f"Analyst emphasis (higher = more weight in your synthesis): "
        f"Technical={technical_w:.0%}, Fundamentals={fund_w:.0%}, News={news_w:.0%}. "
        f"Prioritize findings from higher-weighted analysts when forming your consensus view.\n\n"
        "You MUST produce a ResearchBrief JSON with ALL fields populated:\n"
        "- key_findings: list of Finding objects (one per analyst, include analyst name, "
        "key_point, supporting_data, weight matching the analyst's normalized weight)\n"
        "- consensus_view: 2-3 sentence synthesis of the overall investment thesis\n"
        "- conflicts: list of strings describing any analyst disagreements\n"
        "- data_quality_summary: 1-2 sentence summary of data completeness\n"
        "- analyst_confidences: leave as empty dict (populated by node from actual values)\n"
        "- overall_confidence: float 0.0-1.0, weighted average of analyst confidences\n"
        "- notes: any caveats, flagged concerns, or synthesis limitations"
    )

    def fmt_opt(val: float | None, suffix: str = "") -> str:
        return f"{val:.2f}{suffix}" if val is not None else "N/A"

    if technical_report is None or technical_report.confidence == 0.0:
        technical_section = (
            f"### Technical Analyst — Weight: {technical_w:.0%}\n"
            f"  Status: NO DATA AVAILABLE (confidence: {technical_report.confidence if technical_report else 0.0:.1f})\n"
            f"  Notes: {technical_report.notes if technical_report else 'Report missing'}\n"
        )
    else:
        technical_section = (
            f"### Technical Analyst — Weight: {technical_w:.0%}\n"
            f"  Confidence: {technical_report.confidence:.2f}\n"
            f"  Current Price: {fmt_opt(technical_report.current_price, ' INR')}\n"
            f"  RSI: {fmt_opt(technical_report.rsi)} — {technical_report.rsi_interpretation}\n"
            f"  MACD Signal: {technical_report.macd_signal}\n"
            f"  Bollinger Position: {technical_report.bollinger_position}\n"
            f"  MA Crossover: {technical_report.ma_crossover}\n"
            f"  Trend Assessment: {technical_report.trend_assessment}\n"
            f"  Data Sources: {', '.join(technical_report.data_sources) or 'none'}\n"
            f"  Notes: {technical_report.notes or 'none'}\n"
        )

    if fundamentals_report is None or fundamentals_report.confidence == 0.0:
        fundamentals_section = (
            f"### Fundamentals Analyst — Weight: {fund_w:.0%}\n"
            f"  Status: NO DATA AVAILABLE (confidence: {fundamentals_report.confidence if fundamentals_report else 0.0:.1f})\n"
            f"  Notes: {fundamentals_report.notes if fundamentals_report else 'Report missing'}\n"
        )
    else:
        fundamentals_section = (
            f"### Fundamentals Analyst — Weight: {fund_w:.0%}\n"
            f"  Confidence: {fundamentals_report.confidence:.2f}\n"
            f"  P/E Ratio: {fmt_opt(fundamentals_report.pe_ratio)}x\n"
            f"  P/B Ratio: {fmt_opt(fundamentals_report.pb_ratio)}x\n"
            f"  ROCE: {fmt_opt(fundamentals_report.roce, '%')}\n"
            f"  Debt-to-Equity: {fmt_opt(fundamentals_report.debt_to_equity)}x\n"
            f"  Promoter Holding: {fmt_opt(fundamentals_report.promoter_holding_pct, '%')}\n"
            f"  Quarterly Summary: {fundamentals_report.quarterly_result_summary or 'N/A'}\n"
            f"  Assessment: {fundamentals_report.fundamental_assessment}\n"
            f"  Data Sources: {', '.join(fundamentals_report.data_sources) or 'none'}\n"
            f"  Notes: {fundamentals_report.notes or 'none'}\n"
        )

    if news_report is None or news_report.confidence == 0.0:
        news_section = (
            f"### News Analyst — Weight: {news_w:.0%}\n"
            f"  Status: NO DATA AVAILABLE (confidence: {news_report.confidence if news_report else 0.0:.1f})\n"
            f"  Notes: {news_report.notes if news_report else 'Report missing'}\n"
        )
    else:
        news_section = (
            f"### News Analyst — Weight: {news_w:.0%}\n"
            f"  Confidence: {news_report.confidence:.2f}\n"
            f"  Article Count: {news_report.article_count}\n"
            f"  Sentiment Direction: {news_report.sentiment_direction}\n"
            f"  Top Headlines: {'; '.join(news_report.top_headlines[:3]) or 'none'}\n"
            f"  Key Themes: {', '.join(news_report.key_themes) or 'none'}\n"
            f"  Assessment: {news_report.news_assessment}\n"
            f"  Data Sources: {', '.join(news_report.data_sources) or 'none'}\n"
            f"  Notes: {news_report.notes or 'none'}\n"
        )

    human_content = (
        f"## Research Synthesis Request\n\n"
        f"Synthesize the following three analyst reports into a ResearchBrief.\n\n"
        f"{technical_section}\n"
        f"{fundamentals_section}\n"
        f"{news_section}\n"
        f"### Instructions\n\n"
        f"Produce a ResearchBrief JSON that:\n"
        f"1. Extracts the most important finding from each analyst as a Finding object\n"
        f"2. Sets each Finding's weight to match the analyst's normalized weight above\n"
        f"3. Identifies consensus (where analysts agree) and conflicts (where they disagree)\n"
        f"4. Computes overall_confidence as a weighted average of analyst confidences\n"
        f"5. Leaves analyst_confidences as empty dict (the node will populate it)\n"
        f"6. Flags any NO DATA AVAILABLE analysts as reducing data quality\n"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]


async def research_manager_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    weights = normalize_weights(settings.analyst_weights)

    prompt = build_research_manager_prompt(
        technical_report=state.technical_report,
        fundamentals_report=state.fundamentals_report,
        news_report=state.news_report,
        weights=weights,
    )

    llm = create_llm_client("deep_think", settings)
    structured_llm = llm.with_structured_output(ResearchBrief)

    brief: ResearchBrief
    try:
        brief = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            brief = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            return {
                "research_summary": ResearchBrief(
                    key_findings=[],
                    consensus_view="",
                    conflicts=[],
                    data_quality_summary=(
                        "Research synthesis unavailable — downstream "
                        "consumers should treat overall_confidence as 0.0."
                    ),
                    analyst_confidences={
                        "technical": (
                            state.technical_report.confidence
                            if state.technical_report
                            else 0.0
                        ),
                        "fundamentals": (
                            state.fundamentals_report.confidence
                            if state.fundamentals_report
                            else 0.0
                        ),
                        "news": (
                            state.news_report.confidence
                            if state.news_report
                            else 0.0
                        ),
                    },
                    overall_confidence=0.0,
                    notes=(
                        f"LLM error after retry: {retry_err!r} "
                        f"(initial: {first_err!r})"
                    ),
                )
            }

    brief.analyst_confidences = {
        "technical": state.technical_report.confidence if state.technical_report else 0.0,
        "fundamentals": state.fundamentals_report.confidence if state.fundamentals_report else 0.0,
        "news": state.news_report.confidence if state.news_report else 0.0,
    }

    brief.overall_confidence = sum(
        weights.get(analyst, 0.0) * conf
        for analyst, conf in brief.analyst_confidences.items()
    )

    return {"research_summary": brief}
