from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from financialagents.observability.events import PipelineEvent


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_FRAGMENT_PAGE_PATH = REPO_ROOT / "tests" / "dashboard" / "_live_fragment_app.py"


class TestApplyEventToStageState:
    def test_apply_event_helper(self) -> None:
        from financialagents.dashboard.live_fragment import (
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
        from financialagents.dashboard.live_fragment import (
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
        from financialagents.dashboard.live_fragment import (
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
        from financialagents.dashboard.live_fragment import (
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


class TestPersonaSizePctRender:
    def test_persona_size_pct_shown_in_status_label(self, tmp_path: Path) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required for live region render test",
        ).AppTest

        app_src = (
            "import streamlit as st\n"
            "from financialagents.observability.events import PipelineEvent\n"
            "from financialagents.dashboard.live_fragment import (\n"
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
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )
        markdown_text = " ".join(m.value for m in at.markdown)
        assert "4.5%" in markdown_text, (
            f"Persona size_pct preview '4.5%' missing from rendered output: "
            f"{markdown_text!r}"
        )
