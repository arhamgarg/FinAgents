from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from financialagents.agents.managers.schemas import (
    PortfolioContext,
    PortfolioDecision,
    ResearchBrief,
)
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.graph.reducers import merge_by_key


class InvestDebateState(BaseModel):

    bull_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    bear_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    current_round: int = 0
    max_rounds: int = 3
    research_brief_snapshot: str = ""
    investment_plan: Optional[str] = None

    def bull_turns(self) -> list[str]:
        return [v for _, v in sorted(self.bull_history.items())]

    def bear_turns(self) -> list[str]:
        return [v for _, v in sorted(self.bear_history.items())]


class RiskDebateState(BaseModel):

    aggressive_assessment: Optional[str] = None
    neutral_assessment: Optional[str] = None
    conservative_assessment: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 2
    risk_debate_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    trader_signal_snapshot: str = ""
    synthesized_position_size_pct: Optional[float] = None
    final_risk_decision: Optional[str] = None

    def risk_turns(self) -> list[str]:
        return [v for _, v in sorted(self.risk_debate_history.items())]


class AgentState(BaseModel):

    ticker: str = ""
    date: str = ""

    sector: Optional[str] = None
    selected_tickers: list[str] = Field(default_factory=list)

    technical_report: Optional[TechnicalReport] = None
    news_report: Optional[NewsReport] = None
    fundamentals_report: Optional[FundamentalsAnalystReport] = None

    research_summary: Optional[ResearchBrief] = None

    investment_debate_state: Optional[InvestDebateState] = None
    risk_debate_state: Optional[RiskDebateState] = None

    trader_signal: Optional[TradingSignal] = None
    portfolio_context: Optional[PortfolioContext] = None

    trader_investment_plan: Optional[str] = None

    final_trade_decision: Optional[PortfolioDecision] = None

    memory_context: Optional[str] = None

    messages: Annotated[list[BaseMessage], add] = Field(default_factory=list)
