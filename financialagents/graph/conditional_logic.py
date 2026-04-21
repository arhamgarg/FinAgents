from __future__ import annotations

from financialagents.agents.utils.agent_states import InvestDebateState, RiskDebateState


def invest_debate_router(state: InvestDebateState) -> str:
    if state.current_round >= state.max_rounds:
        return "investment_plan_node"
    return "bull_researcher_node"


def risk_debate_router(state: RiskDebateState) -> str:
    if state.current_round >= state.max_rounds:
        return "END"
    return "aggressive_risk_node"
