from __future__ import annotations

import asyncio
from typing import Any

import pytest




class TestPipelineEventDataclass:
    def test_pipeline_event_is_frozen_and_hashable(self) -> None:
        from financialagents.observability.events import PipelineEvent

        ev = PipelineEvent(
            kind="node_status",
            node_name="trader_node",
            status="completed",
        )
        assert hash(ev) == hash(ev)
        with pytest.raises(Exception):
            ev.kind = "errored"  # type: ignore[misc]

    def test_module_public_surface(self) -> None:
        from financialagents.observability import events as mod

        assert set(mod.__all__) == {
            "PipelineEvent",
            "EventKind",
            "NodeStatus",
            "STAGE_FOR_NODE",
            "stream_pipeline_events",
        }

    def test_stage_for_node_covers_production_graph(self) -> None:
        from financialagents.observability.events import STAGE_FOR_NODE

        production_nodes = {
            "technical_analyst_node": "Analysts",
            "fundamentals_analyst_node": "Analysts",
            "news_analyst_node": "Analysts",
            "research_manager_node": "Research Manager",
            "debate_init_node": "Investment Debate",
            "invest_debate_node": "Investment Debate",
            "bull_researcher_node": "Investment Debate",
            "bear_researcher_node": "Investment Debate",
            "investment_plan_node": "Investment Debate",
            "trader_node": "Trader",
            "risk_init_node": "Risk Debate + PM",
            "risk_debate_node": "Risk Debate + PM",
            "aggressive_risk_node": "Risk Debate + PM",
            "neutral_risk_node": "Risk Debate + PM",
            "conservative_risk_node": "Risk Debate + PM",
            "risk_debate_consolidation_node": "Risk Debate + PM",
            "portfolio_manager_node": "Risk Debate + PM",
        }
        for node_name, expected_stage in production_nodes.items():
            assert STAGE_FOR_NODE[node_name] == expected_stage, (
                f"{node_name} should map to {expected_stage!r}"
            )

        assert set(STAGE_FOR_NODE.values()) == {
            "Analysts",
            "Research Manager",
            "Investment Debate",
            "Trader",
            "Risk Debate + PM",
        }




class _MockGraph:

    def __init__(self, scripted: list[tuple[tuple[str, ...], Any]]) -> None:
        self.scripted = scripted

    async def astream(self, *args: Any, **kwargs: Any):
        from financialagents.observability.events import STAGE_FOR_NODE

        for ns, payload in self.scripted:
            if isinstance(payload, dict) and any(
                k in STAGE_FOR_NODE for k in payload
            ):
                mode = "updates"
            else:
                mode = "values"
            yield ns, mode, payload


class TestPipelineEventStream:
    @pytest.mark.asyncio
    async def test_subgraph_node_names(self) -> None:
        from financialagents.observability.events import (
            STAGE_FOR_NODE,
            stream_pipeline_events,
        )

        g = _MockGraph(
            [
                (("invest_debate:abc",), {"bull_researcher_node": {"bull_history": {0: "arg1"}}}),
            ]
        )
        evs = []
        async for ev in stream_pipeline_events(g, {}, {}):
            evs.append(ev)
        assert STAGE_FOR_NODE["bull_researcher_node"] == "Investment Debate"
        node_evs = [e for e in evs if e.node_name == "bull_researcher_node"]
        assert any(e.status == "completed" for e in node_evs)

    @pytest.mark.asyncio
    async def test_round_counter(self) -> None:
        from financialagents.observability.events import stream_pipeline_events

        g = _MockGraph(
            [
                ((), {"bull_researcher_node": {"bull_history": {0: "a"}}}),
                ((), {"bear_researcher_node": {"bear_history": {0: "b"}}}),
                ((), {"bull_researcher_node": {"bull_history": {0: "a", 1: "a2"}}}),
            ]
        )
        evs = []
        async for ev in stream_pipeline_events(g, {}, {}):
            evs.append(ev)
        advances = [e for e in evs if e.kind == "round_advanced"]
        assert any(e.round == 2 for e in advances)

    @pytest.mark.asyncio
    async def test_persona_size_pct(self) -> None:
        from financialagents.observability.events import stream_pipeline_events

        g_dict = _MockGraph(
            [
                (
                    (),
                    {
                        "aggressive_risk_node": {
                            "aggressive_assessment": {
                                "recommended_position_size_pct": 4.5,
                            }
                        }
                    },
                ),
            ]
        )
        evs = []
        async for ev in stream_pipeline_events(g_dict, {}, {}):
            evs.append(ev)
        agg = [
            e
            for e in evs
            if e.node_name == "aggressive_risk_node" and e.status == "completed"
        ]
        assert agg and agg[0].position_size_pct == 4.5

        g_json = _MockGraph(
            [
                (
                    (),
                    {
                        "aggressive_risk_node": {
                            "aggressive_assessment": (
                                '{"recommended_position_size_pct": 0.045}'
                            )
                        }
                    },
                ),
            ]
        )
        evs = []
        async for ev in stream_pipeline_events(g_json, {}, {}):
            evs.append(ev)
        agg = [
            e
            for e in evs
            if e.node_name == "aggressive_risk_node" and e.status == "completed"
        ]
        assert agg and agg[0].position_size_pct == 0.045

    @pytest.mark.asyncio
    async def test_raises_become_errored_event(self) -> None:
        from financialagents.observability.events import stream_pipeline_events

        class _RaisingGraph:
            async def astream(self, *args: Any, **kwargs: Any):
                if False:
                    yield None  # pragma: no cover
                raise RuntimeError("boom")

        evs = []
        with pytest.raises(RuntimeError):
            async for ev in stream_pipeline_events(_RaisingGraph(), {}, {}):
                evs.append(ev)
        assert any(
            e.kind == "errored" and e.status == "errored" for e in evs
        ), f"expected errored event before re-raise, got {evs!r}"

    @pytest.mark.asyncio
    async def test_completed_event_carries_final_state(self) -> None:
        from financialagents.observability.events import stream_pipeline_events

        full_state = {"ticker": "RELIANCE", "date": "2026-04-26"}
        g = _MockGraph(
            [
                ((), {"trader_node": {"trader_signal": None}}),
                ((), full_state),
            ]
        )
        evs = []
        async for ev in stream_pipeline_events(g, {}, {}):
            evs.append(ev)
        completed = [e for e in evs if e.kind == "completed"]
        assert completed
        assert completed[0].final_state is None or (
            completed[0].final_state.ticker == "RELIANCE"
        )

    @pytest.mark.asyncio
    async def test_cancelled_via_asyncio_cancel(self) -> None:
        from financialagents.observability.events import stream_pipeline_events

        class _SlowGraph:
            async def astream(self, *args: Any, **kwargs: Any):
                yield (), "updates", {"trader_node": {}}
                await asyncio.sleep(10)
                yield (), "values", {}  # pragma: no cover

        evs: list[Any] = []

        async def consume() -> None:
            async for ev in stream_pipeline_events(_SlowGraph(), {}, {}):
                evs.append(ev)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert any(
            e.kind == "errored" and e.status == "cancelled" for e in evs
        ), f"expected cancelled event, got {evs!r}"
