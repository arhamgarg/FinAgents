from __future__ import annotations

import asyncio
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from finagents.agents.utils.agent_states import AgentState
from finagents.config import Settings
from finagents.graph.setup import build_analysis_graph


class FinAgentsGraph:
    def __init__(self, full_pipeline: bool = False) -> None:
        if full_pipeline:
            from finagents.graph.setup import build_full_pipeline  # noqa: PLC0415

            self._graph = build_full_pipeline()
        else:
            self._graph = build_analysis_graph()

    def compiled(self) -> CompiledStateGraph:
        return self._graph

    async def apropagate(
        self,
        ticker: str,
        date: str,
        settings: Settings | None = None,
    ) -> AgentState:
        if settings is None:
            settings = Settings()

        initial_state = AgentState(ticker=ticker, date=date)

        thread_id = f"{ticker}_{date}_{uuid4().hex[:8]}"

        config: RunnableConfig = {
            "configurable": {
                "settings": settings,
                "thread_id": thread_id,
            }
        }

        result = await self._graph.ainvoke(initial_state, config=config)

        if isinstance(result, dict):
            return AgentState.model_validate(result)
        return result

    def propagate(
        self,
        ticker: str,
        date: str,
        settings: Settings | None = None,
    ) -> AgentState:
        return asyncio.run(self.apropagate(ticker, date, settings))
