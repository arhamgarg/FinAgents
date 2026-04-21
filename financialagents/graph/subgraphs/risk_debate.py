from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from financialagents.agents.risk_mgmt.aggressive import aggressive_risk_node
from financialagents.agents.risk_mgmt.conservative import conservative_risk_node
from financialagents.agents.risk_mgmt.consolidation import risk_debate_consolidation_node
from financialagents.agents.risk_mgmt.neutral import neutral_risk_node
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.graph.conditional_logic import risk_debate_router


def build_risk_debate_subgraph():
    builder = StateGraph(RiskDebateState)

    builder.add_node("aggressive_risk_node", aggressive_risk_node)
    builder.add_node("neutral_risk_node", neutral_risk_node)
    builder.add_node("conservative_risk_node", conservative_risk_node)
    builder.add_node("risk_debate_consolidation_node", risk_debate_consolidation_node)

    builder.add_edge(START, "aggressive_risk_node")
    builder.add_edge("aggressive_risk_node", "neutral_risk_node")
    builder.add_edge("neutral_risk_node", "conservative_risk_node")
    builder.add_edge("conservative_risk_node", "risk_debate_consolidation_node")
    builder.add_conditional_edges(
        "risk_debate_consolidation_node",
        risk_debate_router,
        {
            "aggressive_risk_node": "aggressive_risk_node",
            "END": END,
        },
    )

    return builder.compile()
