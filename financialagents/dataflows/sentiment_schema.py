from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

SentimentLabel = Literal["bullish", "bearish", "neutral"]


class RedditPost(BaseModel):
    title: str
    score: int
    url: str
    created_utc: datetime
    subreddit: str
    selftext_preview: Optional[str] = None


class RedditSentiment(BaseModel):
    posts: list[RedditPost] = []
    post_count: int = 0
    avg_score: float = 0.0


class StockTwitsMessage(BaseModel):
    id: int
    body: str
    created_at: datetime
    sentiment: Optional[SentimentLabel] = None


class StockTwitsSentiment(BaseModel):
    messages: list[StockTwitsMessage] = []
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class TrendsSentiment(BaseModel):
    ticker: str
    interest_over_time: list[tuple[datetime, int]] = []
    peak_interest: int = 0
    avg_interest: float = 0.0


class SentimentReport(BaseModel):
    ticker: str
    reddit: RedditSentiment = RedditSentiment()
    stocktwits: StockTwitsSentiment = StockTwitsSentiment()
    trends: TrendsSentiment | None = None
    sources_status: dict[str, str] = {}
