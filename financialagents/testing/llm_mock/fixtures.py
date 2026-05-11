from __future__ import annotations

from pydantic import BaseModel

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from financialagents.agents.managers.schemas import PortfolioDecision, ResearchBrief
from financialagents.agents.researchers.bear_researcher import BearArgument
from financialagents.agents.researchers.bull_researcher import BullArgument
from financialagents.agents.researchers.investment_thesis import InvestmentThesis
from financialagents.agents.risk_mgmt.schemas import (
    PersonaAssessment,
    RiskConsolidation,
)
from financialagents.agents.trader.schemas import TradingSignal



TECH_REPORT_DEFAULT = TechnicalReport(
    confidence=0.8,
    current_price=2500.0,
    atr_14=35.0,
    circuit_band_pct=0.20,
    t2t_flag=False,
)

FUND_REPORT_DEFAULT = FundamentalsAnalystReport(confidence=0.8)

NEWS_REPORT_DEFAULT = NewsReport(confidence=0.7)

BRIEF_DEFAULT = ResearchBrief(
    consensus_view="bullish on RELIANCE",
    overall_confidence=0.75,
)

TRADER_SIGNAL_DEFAULT = TradingSignal(
    action="BUY",
    conviction=0.7,
    entry_price=2500.0,
    stop_loss=2450.0,
    target_1=2600.0,
    target_2=2700.0,
    position_size_pct=0.03,
    position_size_shares=30,
    risk_reward_ratio=2.0,
    rationale="canonical harness trader signal",
    time_horizon="positional",
)

AGGRESSIVE_DEFAULT = PersonaAssessment(
    main_argument="aggressive thesis — go heavier on conviction",
    recommended_position_size_pct=0.05,
    confidence=0.8,
)

NEUTRAL_DEFAULT = PersonaAssessment(
    main_argument="neutral thesis — split the difference",
    counter_to_prior_persona="disagree with aggressive size",
    recommended_position_size_pct=0.03,
    confidence=0.6,
)

CONSERVATIVE_DEFAULT = PersonaAssessment(
    main_argument="conservative thesis — prefer half-size",
    counter_to_prior_persona="prefer smaller exposure than neutral",
    recommended_position_size_pct=0.01,
    confidence=0.9,
)

PM_DECISION_DEFAULT = PortfolioDecision(
    signal=TRADER_SIGNAL_DEFAULT,
    approved=True,
    decision="approve",
    pm_notes=(
        "approved — sector concentration low; cash headroom available; "
        "size matches risk-debate consolidation"
    ),
)



BULL_ARGUMENT_DEFAULT = BullArgument(
    argument="canonical bull argument — fundamentals + technicals align on RELIANCE",
    key_points=[
        "earnings beat + margin expansion",
        "technical: breakout above 200DMA with healthy volume",
    ],
)

BEAR_ARGUMENT_DEFAULT = BearArgument(
    argument=(
        "canonical bear argument — macro headwinds plus valuation stretch warrant caution"
    ),
    key_points=[
        "INR weakness pressures import-heavy segments",
        "trailing P/E above 3y average",
    ],
)

INVESTMENT_THESIS_DEFAULT = InvestmentThesis(
    synthesis=(
        "accumulate on dips with positional horizon; bull case dominates on "
        "fundamentals + technicals; bear risks are macro and acknowledged."
    ),
    bull_strength="bull case held up across rounds — earnings narrative was decisive",
    bear_strength="bear made the valuation point cleanly but did not refute earnings",
)

RISK_CONSOLIDATION_DEFAULT = RiskConsolidation(
    synthesis=(
        "weighted 3% position view — aggressive 5% / neutral 3% / conservative 1% "
        "blended per Settings.risk_persona_weights. Sizing favours moderate exposure."
    ),
)



HAPPY_PATH_DEFAULTS: dict[str, BaseModel] = {
    "technical_analyst": TECH_REPORT_DEFAULT,
    "fundamentals_analyst": FUND_REPORT_DEFAULT,
    "news_analyst": NEWS_REPORT_DEFAULT,
    "research_manager": BRIEF_DEFAULT,
    "bull_researcher": BULL_ARGUMENT_DEFAULT,
    "bear_researcher": BEAR_ARGUMENT_DEFAULT,
    "investment_thesis": INVESTMENT_THESIS_DEFAULT,
    "trader": TRADER_SIGNAL_DEFAULT,
    "aggressive_risk": AGGRESSIVE_DEFAULT,
    "neutral_risk": NEUTRAL_DEFAULT,
    "conservative_risk": CONSERVATIVE_DEFAULT,
    "risk_debate_consolidation": RISK_CONSOLIDATION_DEFAULT,
    "portfolio_manager": PM_DECISION_DEFAULT,
}
