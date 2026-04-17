from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.analysts.schemas import SentimentAnalystReport
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.dataflows.interface import get_sentiment
from financialagents.dataflows.sentiment_schema import SentimentReport
from financialagents.llm_clients.factory import create_llm_client


def build_sentiment_prompt(
    ticker: str, report: SentimentReport
) -> list[BaseMessage]:
    system_content = (
        "You are an expert Indian stock market sentiment analyst specialising in "
        "retail investor social media sentiment for NSE and BSE listed equities.\n\n"
        "Key Indian stock social media platforms:\n"
        "- r/IndianStreetBets: India's largest retail investing subreddit (~800K members). "
        "Tone is speculative; posts about large-caps and mid-caps. High engagement = "
        "strong retail interest.\n"
        "- r/DalalStreetTalks: More analytical Indian investing subreddit. Focuses on "
        "fundamentals and long-term investing. Less noisy than IndianStreetBets.\n"
        "- StockTwits: Real-time micro-sentiment platform. Bullish/bearish tags are "
        "self-reported by users. High bullish/total ratio indicates positive retail sentiment.\n"
        "- Google Trends: Relative search interest (0-100). Spikes often precede "
        "large price moves (either direction).\n\n"
        "Evaluation guidelines:\n"
        "- Retail sentiment is a lagging/confirming indicator in Indian markets\n"
        "- High retail bullishness can indicate crowded trades (contrarian signal at extremes)\n"
        "- Distinguish between informed discussion vs. FOMO-driven hype\n"
        "- avg_score on Reddit posts reflects engagement quality (higher is better)\n"
        "- StockTwits bullish_count / (bullish_count + bearish_count) = sentiment ratio\n\n"
        "combined_sentiment must be one of: 'bullish', 'bearish', 'neutral'.\n\n"
        "You MUST produce a SentimentAnalystReport JSON with ALL fields populated. "
        "Confidence: 1.0 = many posts with clear sentiment, 0.5 = few posts or mixed, "
        "0.0 = no data. "
        "Set data_sources and post_count after you receive the prompt — "
        "they will be populated by the node."
    )

    reddit = report.reddit
    reddit_section = (
        f"  Post Count: {reddit.post_count}\n"
        f"  Average Score: {reddit.avg_score:.1f}\n"
    )
    if reddit.posts:
        top_posts = reddit.posts[:3]
        post_lines = []
        for post in top_posts:
            preview = post.selftext_preview or "(no preview)"
            post_lines.append(
                f"  - [{post.subreddit}] \"{post.title}\" (score: {post.score})\n"
                f"    Preview: {preview}"
            )
        reddit_section += " Top Posts:\n" + "\n".join(post_lines)
    else:
        reddit_section += " No posts available"

    st = report.stocktwits
    total_tagged = st.bullish_count + st.bearish_count + st.neutral_count
    sentiment_ratio = (
        f"{st.bullish_count / total_tagged * 100:.0f}% bullish"
        if total_tagged > 0
        else "N/A"
    )
    stocktwits_section = (
        f"  Total Messages: {len(st.messages)}\n"
        f"  Bullish: {st.bullish_count} | Bearish: {st.bearish_count} | "
        f"Neutral: {st.neutral_count}\n"
        f"  Sentiment Ratio: {sentiment_ratio}\n"
    )
    if st.messages:
        sample = st.messages[:3]
        msg_lines = [
            f"  - [{m.sentiment or 'untagged'}] \"{m.body[:100]}\""
            for m in sample
        ]
        stocktwits_section += " Sample Messages:\n" + "\n".join(msg_lines)

    if report.trends:
        trends_section = (
            f"  Peak Interest: {report.trends.peak_interest}/100\n"
            f"  Average Interest: {report.trends.avg_interest:.1f}/100\n"
        )
    else:
        trends_section = " Trends data not available"

    human_content = (
        f"## Sentiment Analysis Request: {ticker}\n\n"
        f"### Reddit Sentiment (r/IndianStreetBets + r/DalalStreetTalks)\n"
        f"{reddit_section}\n\n"
        f"### StockTwits Sentiment\n"
        f"{stocktwits_section}\n\n"
        f"### Google Trends\n"
        f"{trends_section}\n\n"
        f"### Instructions\n\n"
        f"Based on the above data, produce a complete SentimentAnalystReport JSON with:\n"
        f"- reddit_score: float 0.0-1.0 scoring Reddit sentiment quality "
        f"(consider post_count and avg_score)\n"
        f"- stocktwits_score: float 0.0-1.0 scoring StockTwits bullish ratio\n"
        f"- combined_sentiment: 'bullish', 'bearish', or 'neutral' (weighted combination)\n"
        f"- post_count: leave as 0 (populated by the node)\n"
        f"- sentiment_assessment: 3-4 sentence Indian-market-aware assessment of "
        f"retail sentiment for {ticker}\n"
        f"- confidence: float 0.0-1.0 based on data volume and quality\n"
        f"- data_sources: leave as empty list (populated by the node)\n"
        f"- notes: any caveats, noise signals, or contrarian observations"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]


async def sentiment_analyst_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    result = await get_sentiment(state.ticker, settings=settings)

    if result.status in ("error", "empty") or result.data is None:
        return {
            "sentiment_report": SentimentAnalystReport(
                confidence=0.0,
                data_sources=[],
                notes=f"No sentiment data: {result.error or 'empty'}",
            )
        }

    prompt = build_sentiment_prompt(state.ticker, result.data)

    llm = create_llm_client("quick_think", settings)
    structured_llm = llm.with_structured_output(SentimentAnalystReport)

    report: SentimentAnalystReport
    try:
        report = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            report = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            raise retry_err from first_err

    active_sources = [
        src
        for src, status in result.data.sources_status.items()
        if status != "error"
    ]
    if active_sources:
        report.data_sources = active_sources

    report.post_count = (
        result.data.reddit.post_count + len(result.data.stocktwits.messages)
    )

    return {"sentiment_report": report}
