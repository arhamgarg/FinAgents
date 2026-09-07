from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finagents.observability.events import PipelineEvent


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_FRAGMENT_PAGE_PATH = REPO_ROOT / "tests" / "dashboard" / "_live_fragment_app.py"


class TestApplyEventToStageState:
    def test_apply_event_helper(self) -> None:
        from finagents.dashboard.live_fragment import (
            _initial_stage_state,
            apply_event_to_stage_state,
        )

        stage_state = _initial_stage_state()

        apply_event_to_stage_state(
            stage_state,
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            ),
            now=10.0,
        )
        analysts = stage_state["Analysts"]
        assert analysts["status"] == "running"
        assert analysts["started_at"] == 10.0
        child = analysts["children"]["technical_analyst_node"]
        assert child["status"] == "running"
        assert child["started_at"] == 10.0
        assert child["completed_at"] is None

        apply_event_to_stage_state(
            stage_state,
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            ),
            now=12.5,
        )
        analysts = stage_state["Analysts"]
        child = analysts["children"]["technical_analyst_node"]
        assert child["status"] == "completed"
        assert child["completed_at"] == 12.5
        assert analysts["status"] == "completed"
        assert analysts["completed_at"] == 12.5

    def test_round_advanced_updates_debate_round(self) -> None:
        from finagents.dashboard.live_fragment import (
            _initial_stage_state,
            apply_event_to_stage_state,
        )

        stage_state = _initial_stage_state()
        apply_event_to_stage_state(
            stage_state,
            PipelineEvent(kind="round_advanced", round=2),
            now=20.0,
        )
        assert stage_state["Investment Debate"]["round"] == 2


class TestDrainQueueAndApply:
    def test_drain_queue_processes_events_in_order(self) -> None:
        from finagents.dashboard.live_fragment import (
            _initial_stage_state,
            drain_queue_and_apply,
        )

        q: queue.Queue[Any] = queue.Queue()
        events = [
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            ),
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            ),
            PipelineEvent(
                kind="node_status",
                node_name="fundamentals_analyst_node",
                status="started",
            ),
        ]
        for ev in events:
            q.put(ev)

        stage_state = _initial_stage_state()
        still_running, last_event = drain_queue_and_apply(stage_state, q)

        assert still_running is True
        assert last_event is not None
        assert last_event.node_name == "fundamentals_analyst_node"
        analysts = stage_state["Analysts"]
        assert analysts["children"]["technical_analyst_node"]["status"] == "completed"
        assert analysts["children"]["fundamentals_analyst_node"]["status"] == "running"

    def test_completed_event_sets_session_state_flags(self) -> None:
        from finagents.dashboard.live_fragment import (
            _initial_stage_state,
            drain_queue_and_apply,
        )

        q: queue.Queue[Any] = queue.Queue()
        q.put(
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            )
        )
        q.put(PipelineEvent(kind="completed"))

        stage_state = _initial_stage_state()
        still_running, last_event = drain_queue_and_apply(stage_state, q)

        assert still_running is False
        assert last_event is not None
        assert last_event.kind == "completed"

    def test_errored_event_preserves_error_field(self) -> None:
        from finagents.dashboard.live_fragment import (
            _initial_stage_state,
            drain_queue_and_apply,
        )

        original_exc = RuntimeError("boom: 260526-no0-marker")
        q: queue.Queue[Any] = queue.Queue()
        q.put(
            PipelineEvent(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            )
        )
        q.put(
            PipelineEvent(
                kind="errored",
                status="errored",
                error=original_exc,
            )
        )

        stage_state = _initial_stage_state()
        still_running, last_event = drain_queue_and_apply(stage_state, q)

        assert still_running is False
        assert last_event is not None
        assert last_event.kind == "errored"
        assert last_event.error is original_exc
        assert type(last_event.error) is RuntimeError
        assert "boom" in str(last_event.error)
        assert "260526-no0-marker" in str(last_event.error)


class TestPersonaSizePctRender:
    def test_persona_size_pct_shown_in_status_label(self, tmp_path: Path) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required for live region render test",
        ).AppTest

        app_src = (
            "import streamlit as st\n"
            "from finagents.observability.events import PipelineEvent\n"
            "from finagents.dashboard.live_fragment import (\n"
            " _initial_stage_state,\n"
            " apply_event_to_stage_state,\n"
            " render_live_region,\n"
            ")\n"
            "stage_state = _initial_stage_state()\n"
            "apply_event_to_stage_state(\n"
            " stage_state,\n"
            " PipelineEvent(\n"
            " kind='node_status',\n"
            " node_name='aggressive_risk_node',\n"
            " status='started',\n"
            "),\n"
            " now=1.0,\n"
            ")\n"
            "apply_event_to_stage_state(\n"
            " stage_state,\n"
            " PipelineEvent(\n"
            " kind='node_status',\n"
            " node_name='aggressive_risk_node',\n"
            " status='completed',\n"
            " position_size_pct=0.045,\n"
            "),\n"
            " now=2.0,\n"
            ")\n"
            "st.session_state['stage_state'] = stage_state\n"
            "render_live_region()\n"
        )
        app_path = tmp_path / "live_fragment_render_app.py"
        app_path.write_text(app_src)

        at = AppTest.from_file(str(app_path)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
        markdown_text = " ".join(m.value for m in at.markdown)
        assert "4.5%" in markdown_text, (
            f"Persona size_pct preview '4.5%' missing from rendered output: {markdown_text!r}"
        )




class _RecordingAsyncGen:

    def __init__(self, inner) -> None:
        self._inner = inner
        self.aclose_awaited = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._inner.__anext__()

    async def aclose(self) -> None:  # noqa: D401 — mirror agen.aclose signature
        self.aclose_awaited = True
        try:
            await self._inner.aclose()
        except Exception:  # pragma: no cover - aclose is best-effort
            pass


class _FakeGraph:

    def __init__(self, recording_agen: _RecordingAsyncGen) -> None:
        self._recording_agen = recording_agen

    def astream(self, *_args, **_kwargs):
        return self._recording_agen


class TestCancelPipeline:

    @pytest.mark.asyncio
    async def test_cancel_event_set_mid_stream_terminates_producer(self) -> None:
        from finagents.observability.events import stream_pipeline_events

        async def upstream():
            yield (
                (),
                "updates",
                {"technical_analyst_node": {}},
            )
            yield (
                (),
                "updates",
                {"fundamentals_analyst_node": {}},
            )

        recording = _RecordingAsyncGen(upstream())
        fake_graph = _FakeGraph(recording)
        evt = threading.Event()

        events: list[PipelineEvent] = []
        async for ev in stream_pipeline_events(
            fake_graph, {}, {"configurable": {}}, cancel_event=evt
        ):
            events.append(ev)
            if len(events) == 1:
                evt.set()

        assert events, "no events drained — generator yielded nothing"
        terminal = events[-1]
        assert terminal.kind == "errored", (
            f"expected terminal kind='errored', got {terminal.kind!r}"
        )
        assert terminal.status == "cancelled", (
            f"expected terminal status='cancelled', got {terminal.status!r}"
        )
        assert recording.aclose_awaited, (
            "stream_pipeline_events did not call agen.aclose() on cancel"
        )
        assert all(ev.node_name != "fundamentals_analyst_node" for ev in events), (
            "stream_pipeline_events drained step 2 after cancel — "
            "expected early return before __anext__"
        )

    def test_cancel_pipeline_helper_flips_flag_and_joins_thread(self) -> None:
        from finagents.dashboard.live_fragment import cancel_pipeline

        class _SessionStateShim(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

            def __setattr__(self, key, value) -> None:
                self[key] = value

            def __delattr__(self, key) -> None:
                try:
                    del self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

        evt = threading.Event()
        thread = threading.Thread(
            target=evt.wait,
            daemon=True,
        )
        thread.start()

        try:
            shim = _SessionStateShim()
            shim["pipeline_cancel"] = evt
            shim["pipeline_thread"] = thread
            shim["pipeline_running"] = True

            import streamlit as st
            from unittest.mock import patch

            with patch.object(st, "session_state", shim):
                cancel_pipeline()

            assert evt.is_set(), "cancel_pipeline did not set cancel event"
            assert shim.get("pipeline_running") is False, (
                "cancel_pipeline did not flip pipeline_running to False; "
                f"got {shim.get('pipeline_running')!r}"
            )
            assert shim.get("pipeline_thread") is None, (
                "cancel_pipeline did not clear pipeline_thread"
            )
            assert shim.get("pipeline_cancel") is None, (
                "cancel_pipeline did not clear pipeline_cancel"
            )
        finally:
            evt.set()
            thread.join(timeout=1)
            assert not thread.is_alive()

    def test_start_pipeline_stores_cancel_and_thread_in_session_state(self, monkeypatch) -> None:
        from finagents.dashboard import live_fragment

        monkeypatch.setattr(
            live_fragment,
            "_run_in_background",
            lambda *args, **kwargs: None,
        )

        class _SessionStateShim(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

            def __setattr__(self, key, value) -> None:
                self[key] = value

        shim = _SessionStateShim()
        import streamlit as st
        from unittest.mock import patch

        with patch.object(st, "session_state", shim):
            live_fragment.start_pipeline(MagicMock(), {}, {"configurable": {}})

        assert isinstance(shim.get("pipeline_cancel"), threading.Event), (
            "start_pipeline did not seed pipeline_cancel as threading.Event; "
            f"got {type(shim.get('pipeline_cancel'))!r}"
        )
        assert isinstance(shim.get("pipeline_thread"), threading.Thread), (
            "start_pipeline did not seed pipeline_thread as threading.Thread; "
            f"got {type(shim.get('pipeline_thread'))!r}"
        )
        assert shim.get("pipeline_running") is True
        assert shim.get("pipeline_queue") is not None
