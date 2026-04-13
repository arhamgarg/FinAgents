from __future__ import annotations

import asyncio

from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.graph.setup import build_analysis_graph


class FinancialAgentsGraph:

    def __init__(self) -> None:
        self._graph = build_analysis_graph()

    async def apropagate(
        self,
        ticker: str,
        date: str,
        settings: Settings | None = None,
    ) -> AgentState:
        if settings is None:
            settings = Settings()

        initial_state = AgentState(ticker=ticker, date=date)
        config = {"configurable": {"settings": settings}}

        result = await self._graph.ainvoke(initial_state, config=config)

        if isinstance(result, dict):
            return AgentState(**result)
        return result

    def propagate(
        self,
        ticker: str,
        date: str,
        settings: Settings | None = None,
    ) -> AgentState:
        return asyncio.run(self.apropagate(ticker, date, settings))
