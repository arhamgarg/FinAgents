from __future__ import annotations

from financialagents.agents.utils.agent_states import InvestDebateState


def invest_debate_router(state: InvestDebateState) -> str:
    if state.current_round >= state.max_rounds:
        return "investment_plan_node"
    return "bull_researcher_node"
