from __future__ import annotations

from pydantic import BaseModel, Field


class MarketReport(BaseModel):

    rsi: float | None = None
    rsi_interpretation: str = ""
    macd_signal: str = ""
    macd_line: float | None = None
    macd_hist: float | None = None
    bollinger_position: str = ""
    bb_upper: float | None = None
    bb_lower: float | None = None
    ma_crossover: str = ""
    sma_50: float | None = None
    sma_200: float | None = None
    ema_20: float | None = None
    current_price: float | None = None
    trend_assessment: str = ""
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_sources: list[str] = Field(default_factory=list)
    notes: str = ""


class FundamentalsAnalystReport(BaseModel):

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    roce: float | None = None
    debt_to_equity: float | None = None
    quarterly_result_summary: str = ""
    promoter_holding_pct: float | None = None
    fundamental_assessment: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_sources: list[str] = Field(default_factory=list)
    notes: str = ""


class NewsReport(BaseModel):

    article_count: int = 0
    sentiment_direction: str = ""
    top_headlines: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)
    news_assessment: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_sources: list[str] = Field(default_factory=list)
    notes: str = ""


class SentimentAnalystReport(BaseModel):

    reddit_score: float = 0.0
    stocktwits_score: float = 0.0
    combined_sentiment: str = ""
    post_count: int = 0
    sentiment_assessment: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_sources: list[str] = Field(default_factory=list)
    notes: str = ""
