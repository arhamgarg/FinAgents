from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stderr
from typing import Any, AsyncGenerator, AsyncIterator, cast

import pytest


class _SlowFakeGraph:

    async def astream(
        self,
        initial_state: Any,
        *,
        config: Any = None,
        stream_mode: Any = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[tuple[Any, str, Any]]:
        yield ((), "updates", {"trader_node": {"trader_investment_thesis": "BUY"}})
        await asyncio.Event().wait()


class _RaisingFakeGraph:

    async def astream(
        self,
        initial_state: Any,
        *,
        config: Any = None,
        stream_mode: Any = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[tuple[Any, str, Any]]:
        yield ((), "updates", {"trader_node": {"trader_investment_thesis": "BUY"}})
        raise RuntimeError("simulated node failure")


class TestGeneratorExitTeardown:
    async def test_aclose_after_consumer_break_does_not_warn(self) -> None:
        from finagents.observability.events import (
            PipelineEvent,
            stream_pipeline_events,
        )

        gen = cast(
            AsyncGenerator[PipelineEvent, None],
            stream_pipeline_events(
                _SlowFakeGraph(),
                initial_state={"ticker": "TEST", "date": "2026-05-02"},
                config={},
            ),
        )

        events: list[PipelineEvent] = []

        async for ev in gen:
            events.append(ev)
            if ev.kind == "node_status" and ev.status == "completed":
                break

        assert events, "expected events before break; got none"

        await gen.aclose()

        assert all(ev.kind != "errored" for ev in events), (
            "Teardown exception was swallowed and synthesized as 'errored' — "
            "the bug is back. Expected aclose() to propagate the throw "
            f"untouched. Events seen: {events}"
        )

    def test_asyncio_run_shutdown_emits_no_generator_exit_warning(self) -> None:
        from finagents.observability.events import (
            PipelineEvent,
            stream_pipeline_events,
        )

        async def consume() -> None:
            async for ev in stream_pipeline_events(
                _SlowFakeGraph(),
                initial_state={"ticker": "TEST", "date": "2026-05-02"},
                config={},
            ):
                if (
                    isinstance(ev, PipelineEvent)
                    and ev.kind == "node_status"
                    and ev.status == "completed"
                ):
                    break

        captured = io.StringIO()
        with redirect_stderr(captured):
            asyncio.run(consume())

        stderr_text = captured.getvalue()
        sys.stderr.write(stderr_text)
        sys.stderr.flush()

        assert "GeneratorExit" not in stderr_text, (
            "asyncio.run() shutdown produced a 'GeneratorExit' warning — "
            f"the bug is back. stderr was:\n{stderr_text}"
        )
        assert "ignored GeneratorExit" not in stderr_text, (
            "asyncio.run() shutdown produced 'ignored GeneratorExit' — "
            f"the bug is back. stderr was:\n{stderr_text}"
        )
        assert not ("RuntimeError" in stderr_text and "async generator" in stderr_text), (
            "asyncio.run() shutdown produced an async-generator RuntimeError "
            f"— the bug is back. stderr was:\n{stderr_text}"
        )

    async def test_baseexception_path_still_synthesizes_errored(self) -> None:
        from finagents.observability.events import (
            PipelineEvent,
            stream_pipeline_events,
        )

        gen = stream_pipeline_events(
            _RaisingFakeGraph(),
            initial_state={"ticker": "TEST", "date": "2026-05-02"},
            config={},
        )

        events: list[PipelineEvent] = []
        with pytest.raises(RuntimeError, match="simulated node failure"):
            async for ev in gen:
                events.append(ev)

        assert events, "expected at least one event before the raise"
        assert events[-1].kind == "errored", (
            f"expected terminal event kind='errored'; got {events[-1]}"
        )
        assert events[-1].status == "errored"
        assert isinstance(events[-1].error, RuntimeError)
