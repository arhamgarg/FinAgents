from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from finagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from finagents.agents.managers.schemas import PortfolioDecision, ResearchBrief
from finagents.agents.researchers.bear_researcher import BearArgument
from finagents.agents.researchers.bull_researcher import BullArgument
from finagents.agents.researchers.investment_thesis import InvestmentThesis
from finagents.agents.risk_mgmt.schemas import (
    PersonaAssessment,
    RiskConsolidation,
)
from finagents.agents.trader.schemas import TradingSignal

CallSite = Literal[
    "technical_analyst",
    "fundamentals_analyst",
    "news_analyst",
    "research_manager",
    "bull_researcher",
    "bear_researcher",
    "investment_thesis",
    "trader",
    "aggressive_risk",
    "neutral_risk",
    "conservative_risk",
    "risk_debate_consolidation",
    "portfolio_manager",
]


CALL_SITE_SCHEMAS: dict[CallSite, type[BaseModel]] = {
    "technical_analyst": TechnicalReport,
    "fundamentals_analyst": FundamentalsAnalystReport,
    "news_analyst": NewsReport,
    "research_manager": ResearchBrief,
    "bull_researcher": BullArgument,
    "bear_researcher": BearArgument,
    "investment_thesis": InvestmentThesis,
    "trader": TradingSignal,
    "aggressive_risk": PersonaAssessment,
    "neutral_risk": PersonaAssessment,
    "conservative_risk": PersonaAssessment,
    "risk_debate_consolidation": RiskConsolidation,
    "portfolio_manager": PortfolioDecision,
}
