from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from financialagents.agents.researchers.bear_researcher import bear_researcher_node
from financialagents.agents.researchers.bull_researcher import bull_researcher_node
from financialagents.agents.researchers.investment_thesis import investment_thesis_node
from financialagents.agents.utils.agent_states import InvestDebateState
from financialagents.graph.conditional_logic import invest_debate_router


def build_invest_debate_subgraph():
    builder = StateGraph(InvestDebateState)

    builder.add_node("bull_researcher_node", bull_researcher_node)
    builder.add_node("bear_researcher_node", bear_researcher_node)
    builder.add_node("investment_thesis_node", investment_thesis_node)

    builder.add_edge(START, "bull_researcher_node")
    builder.add_edge("bull_researcher_node", "bear_researcher_node")
    builder.add_conditional_edges(
        "bear_researcher_node",
        invest_debate_router,
        {
            "bull_researcher_node": "bull_researcher_node",
            "investment_thesis_node": "investment_thesis_node",
        },
    )
    builder.add_edge("investment_thesis_node", END)

    return builder.compile()
