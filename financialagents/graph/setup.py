from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from financialagents.agents.analysts.fundamentals_analyst import (
    fundamentals_analyst_node,
)
from financialagents.agents.analysts.technical_analyst import technical_analyst_node
from financialagents.agents.analysts.news_analyst import news_analyst_node
from financialagents.agents.analysts.sentiment_analyst import sentiment_analyst_node
from financialagents.agents.managers.research_manager import research_manager_node
from financialagents.agents.utils.agent_states import AgentState, InvestDebateState
from financialagents.config import Settings


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


def debate_init_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    max_rounds = max(1, settings.max_debate_rounds)
    research_brief_text = (
        state.research_summary.consensus_view if state.research_summary else ""
    )

    return {
        "investment_debate_state": InvestDebateState(
            bull_history={},
            bear_history={},
            current_round=0,
            max_rounds=max_rounds,
            research_brief_snapshot=research_brief_text,
        )
    }



_INVEST_DEBATE_SUBGRAPH = None  # type: ignore[var-annotated]


def _reset_invest_debate_subgraph_cache() -> None:
    global _INVEST_DEBATE_SUBGRAPH
    _INVEST_DEBATE_SUBGRAPH = None


async def call_invest_debate_subgraph(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    global _INVEST_DEBATE_SUBGRAPH
    if _INVEST_DEBATE_SUBGRAPH is None:
        from financialagents.graph.subgraphs.invest_debate import (
            build_invest_debate_subgraph,
        )

        _INVEST_DEBATE_SUBGRAPH = build_invest_debate_subgraph()

    debate_state = state.investment_debate_state
    assert debate_state is not None, (
        "debate_init_node must run before invest_debate_subgraph "
        "(state.investment_debate_state is None)"
    )

    recursion_limit = debate_state.max_rounds * 3 + 10

    result = await _INVEST_DEBATE_SUBGRAPH.ainvoke(
        debate_state,
        config={"recursion_limit": recursion_limit},
    )
    updated_debate = InvestDebateState.model_validate(result)
    return {"investment_debate_state": updated_debate}


def build_full_pipeline():
    builder = StateGraph(AgentState)

    builder.add_node("propagation_node", propagation_node)
    builder.add_node("technical_analyst_node", technical_analyst_node)
    builder.add_node("fundamentals_analyst_node", fundamentals_analyst_node)
    builder.add_node("news_analyst_node", news_analyst_node)
    builder.add_node("sentiment_analyst_node", sentiment_analyst_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("research_manager_node", research_manager_node)

    builder.add_node("debate_init_node", debate_init_node)
    builder.add_node("invest_debate_node", call_invest_debate_subgraph)

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

    builder.add_edge("research_manager_node", "debate_init_node")
    builder.add_edge("debate_init_node", "invest_debate_node")

    from financialagents.agents.trader.trader import trader_node as _trader_node

    builder.add_node("trader_node", _trader_node)
    builder.add_edge("invest_debate_node", "trader_node")
    builder.add_edge("trader_node", END)

    return builder.compile()


def _reset_trader_node_cache() -> None:
    return None
