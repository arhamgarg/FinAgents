from __future__ import annotations

from typing import Any, AsyncIterator


async def _fake_astream_three_tuples() -> AsyncIterator[tuple[Any, str, Any]]:
    yield ((), "values", {"ticker": "TEST", "date": "2026-04-28"})
    yield ((), "updates", {"trader_node": {"trader_investment_thesis": "BUY"}})
    yield (
        (),
        "values",
        {"ticker": "TEST", "date": "2026-04-28", "trader_investment_thesis": "BUY"},
    )


class _FakeGraph:

    def astream(
        self,
        initial_state: Any,
        *,
        config: Any = None,
        stream_mode: Any = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[tuple[Any, str, Any]]:
        return _fake_astream_three_tuples()


class TestAstreamThreeTupleUnpack:
    async def test_drains_three_tuple_stream_without_value_error(self) -> None:
        from finagents.observability.events import (
            PipelineEvent,
            stream_pipeline_events,
        )

        fake_graph = _FakeGraph()
        events: list[PipelineEvent] = []

        async for ev in stream_pipeline_events(
            fake_graph,
            initial_state={"ticker": "TEST", "date": "2026-04-28"},
            config={},
        ):
            events.append(ev)


        values_events = [e for e in events if e.kind == "values" and e.final_state is not None]
        assert values_events, (
            "expected at least one kind='values' event with final_state populated; "
            f"got events={events}"
        )

        trader_started = [
            e
            for e in events
            if e.kind == "node_status" and e.node_name == "trader_node" and e.status == "started"
        ]
        assert len(trader_started) == 1, (
            f"expected exactly one trader_node started event; got {trader_started}"
        )

        trader_completed = [
            e
            for e in events
            if e.kind == "node_status" and e.node_name == "trader_node" and e.status == "completed"
        ]
        assert len(trader_completed) == 1, (
            f"expected exactly one trader_node completed event; got {trader_completed}"
        )

        assert events[-1].kind == "completed", (
            f"expected terminal event kind='completed'; got {events[-1]}"
        )
