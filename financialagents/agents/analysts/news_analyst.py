from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.analysts.schemas import NewsReport
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.dataflows.interface import get_news
from financialagents.dataflows.news_schema import NewsArticle
from financialagents.llm_clients.factory import create_llm_client


def build_news_prompt(
    ticker: str, articles: list[NewsArticle]
) -> list[BaseMessage]:
    system_content = (
        "You are an expert Indian financial news analyst specialising in NSE and BSE "
        "listed equities. You analyse news from major Indian financial publications.\n\n"
        "Key Indian financial news sources:\n"
        "- MoneyControl: India's largest financial news portal; covers corporate results, "
        "deal announcements, and market movements\n"
        "- Economic Times (ET): Leading business newspaper; strong on macro, policy, "
        "and corporate earnings\n"
        "- Livemint: Premium financial daily; strong on corporate governance, "
        "M&A, and capital markets\n\n"
        "Evaluate news with Indian market context:\n"
        "- Corporate actions: dividends, buybacks, bonus issues, rights issues\n"
        "- Regulatory news: SEBI orders, RBI policy impact, sector-specific regulations\n"
        "- Sector news: IT deal wins, pharma USFDA approvals, banking NPA disclosures, "
        "auto sales data\n"
        "- Management commentary: guidance upgrades/downgrades, capex announcements\n"
        "- Promoter activity: insider buys/sells, pledge creations/releases\n\n"
        "Sentiment direction must be one of: 'positive', 'negative', 'neutral', 'mixed'.\n\n"
        "You MUST produce a NewsReport JSON with ALL fields populated. "
        "Confidence reflects news freshness and volume: 1.0 = many recent relevant articles, "
        "0.5 = few or stale articles, 0.0 = no articles.\n"
        "Set data_sources and article_count after you receive the prompt — "
        "they will be populated by the node."
    )

    article_lines = []
    for i, article in enumerate(articles, 1):
        ts = article.published_at.strftime("%Y-%m-%d %H:%M UTC") if article.published_at else "unknown date"
        snippet = article.snippet or "(no snippet)"
        article_lines.append(
            f"{i}. [{article.source}] {article.title}\n"
            f"   Published: {ts}\n"
            f"   URL: {article.url}\n"
            f"   Snippet: {snippet}"
        )

    articles_block = "\n\n".join(article_lines)

    human_content = (
        f"## News Analysis Request: {ticker}\n\n"
        f"**Total Articles:** {len(articles)}\n\n"
        f"### Articles\n\n"
        f"{articles_block}\n\n"
        f"### Instructions\n\n"
        f"Based on the {len(articles)} articles above, produce a complete NewsReport JSON with:\n"
        f"- article_count: leave as 0 (populated by the node to {len(articles)})\n"
        f"- sentiment_direction: overall sentiment — 'positive', 'negative', 'neutral', or 'mixed'\n"
        f"- top_headlines: list of the top 3 most impactful headlines (exact titles from above)\n"
        f"- key_themes: list of 2-4 key themes (e.g., 'quarterly results', 'deal wins', "
        f"'regulatory risk', 'management guidance')\n"
        f"- news_assessment: 3-4 sentence synthesis of the news sentiment for {ticker} "
        f"in Indian market context\n"
        f"- confidence: float 0.0-1.0 based on article freshness and relevance\n"
        f"- data_sources: leave as empty list (populated by the node)\n"
        f"- notes: any caveats or observations about the news coverage"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]


async def news_analyst_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    result = await get_news(state.ticker)

    if (
        result.status in ("error", "empty")
        or result.data is None
        or len(result.data) == 0
    ):
        return {
            "news_report": NewsReport(
                confidence=0.0,
                data_sources=[],
                notes=f"No news data: {result.error or 'empty'}",
            )
        }

    prompt = build_news_prompt(state.ticker, result.data)

    llm = create_llm_client("quick_think", settings)
    structured_llm = llm.with_structured_output(NewsReport)

    report: NewsReport
    try:
        report = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            report = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            return {
                "news_report": NewsReport(
                    confidence=0.0,
                    data_sources=[result.source] if result.source else [],
                    article_count=len(result.data),
                    notes=(
                        f"LLM error after retry: {retry_err!r} "
                        f"(initial: {first_err!r})"
                    ),
                )
            }

    if result.source and result.source not in report.data_sources:
        report.data_sources = list(report.data_sources) + [result.source]
    report.article_count = len(result.data)

    n = report.article_count
    if n == 0:
        report.confidence = 0.0
    elif n <= 2:
        report.confidence = 0.3
    elif n <= 5:
        report.confidence = 0.6
    else:
        report.confidence = 1.0

    return {"news_report": report}
