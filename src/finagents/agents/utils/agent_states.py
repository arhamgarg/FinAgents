from __future__ import annotations

from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from finagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from finagents.agents.managers.schemas import (
    PortfolioContext,
    PortfolioDecision,
    ResearchBrief,
)
from finagents.agents.trader.schemas import TradingSignal
from finagents.agents.utils.reducers import merge_by_key


class InvestDebateState(BaseModel):
    bull_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    bear_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    current_round: int = 0
    max_rounds: int = 3
    research_brief_snapshot: str = ""
    investment_thesis: str | None = None

    def bull_turns(self) -> list[str]:
        return [v for _, v in sorted(self.bull_history.items())]

    def bear_turns(self) -> list[str]:
        return [v for _, v in sorted(self.bear_history.items())]


class RiskDebateState(BaseModel):
    aggressive_assessment: str | None = None
    neutral_assessment: str | None = None
    conservative_assessment: str | None = None
    current_round: int = 0
    max_rounds: int = 2
    risk_debate_history: Annotated[dict[int, str], merge_by_key] = Field(default_factory=dict)
    trader_signal_snapshot: str = ""
    synthesized_position_size_pct: float | None = None
    final_risk_decision: str | None = None

    def risk_turns(self) -> list[str]:
        return [v for _, v in sorted(self.risk_debate_history.items())]


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = ""
    date: str = ""

    sector: str | None = None
    selected_tickers: list[str] = Field(default_factory=list)

    technical_report: TechnicalReport | None = None
    news_report: NewsReport | None = None
    fundamentals_report: FundamentalsAnalystReport | None = None

    research_summary: ResearchBrief | None = None

    investment_debate_state: InvestDebateState | None = None
    risk_debate_state: RiskDebateState | None = None

    trader_signal: TradingSignal | None = None
    portfolio_context: PortfolioContext | None = None

    trader_investment_thesis: str | None = None

    final_trade_decision: PortfolioDecision | None = None

    memory_context: str | None = None

    messages: Annotated[list[BaseMessage], add] = Field(default_factory=list)
