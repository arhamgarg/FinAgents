from __future__ import annotations

from operator import add
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class InvestDebateState(BaseModel):

    bull_history: Annotated[list[str], add] = Field(default_factory=list)
    bear_history: Annotated[list[str], add] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 3
    investment_plan: Optional[str] = None


class RiskDebateState(BaseModel):

    aggressive_assessment: Optional[str] = None
    neutral_assessment: Optional[str] = None
    conservative_assessment: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 2
    risk_debate_history: Annotated[list[str], add] = Field(default_factory=list)
    final_risk_decision: Optional[str] = None


class AgentState(BaseModel):

    ticker: str = ""
    date: str = ""

    sector: Optional[str] = None
    selected_tickers: list[str] = Field(default_factory=list)

    market_report: Optional[str] = None
    news_report: Optional[str] = None
    sentiment_report: Optional[str] = None
    fundamentals_report: Optional[str] = None

    research_summary: Optional[str] = None

    investment_debate_state: Optional[InvestDebateState] = None
    risk_debate_state: Optional[RiskDebateState] = None

    trader_investment_plan: Optional[str] = None

    final_trade_decision: Optional[dict[str, Any]] = None

    memory_context: Optional[str] = None

    messages: Annotated[list[BaseMessage], add] = Field(default_factory=list)
