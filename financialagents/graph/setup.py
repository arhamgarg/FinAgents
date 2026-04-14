from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from financialagents.agents.analysts.fundamentals_analyst import (
    fundamentals_analyst_node,
)
from financialagents.agents.analysts.technical_analyst import technical_analyst_node
from financialagents.agents.analysts.news_analyst import news_analyst_node
from financialagents.agents.analysts.sentiment_analyst import sentiment_analyst_node
from financialagents.agents.managers.research_manager import research_manager_node
from financialagents.agents.utils.agent_states import AgentState


def propagation_node(state: AgentState) -> dict:
    return {}


def validation_node(state: AgentState) -> dict:
    missing: list[str] = []
    if state.technical_report is None:
        missing.append("technical_report")
    if state.fundamentals_report is None:
        missing.append("fundamentals_report")
    if state.news_report is None:
        missing.append("news_report")
    if state.sentiment_report is None:
        missing.append("sentiment_report")
    if missing:
        raise ValueError(
            f"Pipeline validation failed: missing analyst reports: {missing}. "
            "Check analyst node logs for errors."
        )
    return {}


def build_analysis_graph():
    builder = StateGraph(AgentState)

    builder.add_node("propagation_node", propagation_node)
    builder.add_node("technical_analyst_node", technical_analyst_node)
    builder.add_node("fundamentals_analyst_node", fundamentals_analyst_node)
    builder.add_node("news_analyst_node", news_analyst_node)
    builder.add_node("sentiment_analyst_node", sentiment_analyst_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("research_manager_node", research_manager_node)

    builder.add_edge(START, "propagation_node")

    builder.add_edge("propagation_node", "technical_analyst_node")
    builder.add_edge("propagation_node", "fundamentals_analyst_node")
    builder.add_edge("propagation_node", "news_analyst_node")
    builder.add_edge("propagation_node", "sentiment_analyst_node")

    builder.add_edge("technical_analyst_node", "validation_node")
    builder.add_edge("fundamentals_analyst_node", "validation_node")
    builder.add_edge("news_analyst_node", "validation_node")
    builder.add_edge("sentiment_analyst_node", "validation_node")

    builder.add_edge("validation_node", "research_manager_node")
    builder.add_edge("research_manager_node", END)

    return builder.compile()
