from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest


class _MockGraph:
    async def astream(
        self,
        input_state: Any,
        *,
        config: Any | None = None,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[tuple[tuple[str, ...], Any]]:
        yield ((), {"technical_analyst_node": {"report": {"action": "BUY"}}})
        yield ((), {"fundamentals_analyst_node": {"report": {"action": "BUY"}}})
        yield ((), {"news_analyst_node": {"report": {"action": "HOLD"}}})
        yield (
            ("invest_debate:abc",),
            {"bull_researcher_node": {"bull_history": {0: "bull turn 0"}}},
        )
        yield (
            ("invest_debate:abc",),
            {"bear_researcher_node": {"bear_history": {0: "bear turn 0"}}},
        )
        yield (
            ("risk_debate:def",),
            {
                "aggressive_risk_node": {
                    "aggressive_assessment": ('{"recommended_position_size_pct": 0.045}')
                }
            },
        )
        yield ((), {"portfolio_manager_node": {"final_trade_decision": None}})


@pytest.fixture
def mock_graph() -> _MockGraph:
    return _MockGraph()
