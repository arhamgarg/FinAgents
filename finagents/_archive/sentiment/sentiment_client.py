from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from finagents.config import Settings
from finagents.dataflows import rate_limits
from finagents.dataflows._ticker import sanitize_ticker
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.sentiment_schema import (
    RedditPost,
    RedditSentiment,
    SentimentReport,
    StockTwitsMessage,
    StockTwitsSentiment,
    TrendsSentiment,
)

STOCKTWITS_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

SOURCE = "sentiment"


async def _fetch_reddit(settings: Settings, ticker: str) -> RedditSentiment:
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        raise ValueError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")

    def _blocking_fetch() -> list[RedditPost]:
        import praw

        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        collected: list[RedditPost] = []
        for subreddit_name in ["IndianStreetBets", "DalalStreetTalks"]:
            subreddit = reddit.subreddit(subreddit_name)
            for post in subreddit.search(ticker, sort="new", time_filter="week", limit=25):
                collected.append(
                    RedditPost(
                        title=getattr(post, "title", ""),
                        score=int(getattr(post, "score", 0)),
                        url=f"https://reddit.com{getattr(post, 'permalink', '')}",
                        created_utc=datetime.fromtimestamp(
                            float(getattr(post, "created_utc", 0)),
                            tz=timezone.utc,
                        ),
                        subreddit=subreddit_name,
                        selftext_preview=((getattr(post, "selftext", "") or "")[:500] or None),
                    )
                )
        return collected

    posts = await asyncio.to_thread(_blocking_fetch)
    avg_score = (sum(p.score for p in posts) / len(posts)) if posts else 0.0
    return RedditSentiment(posts=posts, post_count=len(posts), avg_score=avg_score)


async def _fetch_stocktwits(client: httpx.AsyncClient, ticker: str) -> StockTwitsSentiment:
    await rate_limits.acquire("api.stocktwits.com")
    url = STOCKTWITS_STREAM_URL.format(symbol=ticker)
    response = await client.get(url)
    response.raise_for_status()
    payload = response.json()

    messages_raw = payload.get("messages", []) or []
    messages: list[StockTwitsMessage] = []
    bullish = bearish = neutral = 0
    for m in messages_raw:
        entities = m.get("entities") or {}
        sentiment_field = entities.get("sentiment") if isinstance(entities, dict) else None
        basic: Optional[str] = None
        if isinstance(sentiment_field, dict):
            basic = sentiment_field.get("basic")
        label = None
        if basic == "Bullish":
            label = "bullish"
            bullish += 1
        elif basic == "Bearish":
            label = "bearish"
            bearish += 1
        else:
            neutral += 1
        try:
            created = datetime.fromisoformat(m.get("created_at", "").replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created = datetime.now(timezone.utc)
        messages.append(
            StockTwitsMessage(
                id=int(m.get("id", 0)),
                body=str(m.get("body", "")),
                created_at=created,
                sentiment=label,
            )
        )
    return StockTwitsSentiment(
        messages=messages,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
    )


async def _fetch_trends(ticker: str) -> TrendsSentiment:
    await rate_limits.acquire("trends.google.com")

    def _blocking_trends() -> list[tuple[datetime, int]]:
        from trendspyg import TrendReq  # type: ignore[import-not-found]

        pytrends = TrendReq(hl="en-IN", tz=330)
        pytrends.build_payload([ticker], cat=0, timeframe="today 3-m", geo="IN")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return []
        series: list[tuple[datetime, int]] = []
        for idx, row in df.iterrows():
            try:
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                score = int(row[ticker])
                series.append((ts, score))
            except (KeyError, ValueError, TypeError):
                continue
        return series

    series = await asyncio.to_thread(_blocking_trends)
    if not series:
        return TrendsSentiment(ticker=ticker, interest_over_time=[])
    peak = max(s[1] for s in series)
    avg = sum(s[1] for s in series) / len(series)
    return TrendsSentiment(
        ticker=ticker,
        interest_over_time=series,
        peak_interest=peak,
        avg_interest=avg,
    )


async def get_sentiment(
    ticker: str,
    settings: Optional[Settings] = None,
) -> DataResult[SentimentReport]:
    try:
        safe_ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        return DataResult[SentimentReport](status="error", source=SOURCE, error=str(exc))
    settings = settings or Settings()  # type: ignore[call-arg]

    report = SentimentReport(ticker=safe_ticker)
    sources_status: dict[str, str] = {}

    try:
        reddit_result = await _fetch_reddit(settings, safe_ticker)
        report.reddit = reddit_result
        sources_status["reddit"] = "ok" if reddit_result.posts else "empty"
    except ValueError as exc:
        sources_status["reddit"] = "empty"
        report.sources_status["reddit_error"] = str(exc)
    except Exception as exc:
        sources_status["reddit"] = "error"
        report.sources_status["reddit_error"] = str(exc)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=15.0, connect=5.0)) as client:
            st_result = await _fetch_stocktwits(client, safe_ticker)
        report.stocktwits = st_result
        sources_status["stocktwits"] = "ok"
    except Exception as exc:
        sources_status["stocktwits"] = "error"
        report.sources_status["stocktwits_error"] = str(exc)

    try:
        trends_result = await _fetch_trends(safe_ticker)
        report.trends = trends_result
        sources_status["trends"] = "ok" if trends_result.interest_over_time else "empty"
    except Exception as exc:
        sources_status["trends"] = "error"
        report.sources_status["trends_error"] = str(exc)

    report.sources_status.update(sources_status)

    any_data = (
        bool(report.reddit.posts)
        or bool(report.stocktwits.messages)
        or bool(report.trends and report.trends.interest_over_time)
    )
    if any_data:
        overall = "ok"
    else:
        overall = "empty"

    return DataResult[SentimentReport](status=overall, data=report, source=SOURCE)
