from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any, Literal, Optional

import streamlit as st

from financialagents.observability.events import (
    PipelineEvent,
    STAGE_FOR_NODE,
)


STAGE_ORDER: list[str] = [
    "Analysts",
    "Research Manager",
    "Investment Debate",
    "Trader",
    "Risk Debate + PM",
]

STATUS_ICON: dict[str, str] = {
    "running": ":material/progress_activity:",
    "completed": ":material/check_circle:",
    "errored": ":material/error:",
    "cancelled": ":material/cancel:",
    "pending": ":material/circle:",
}


def _initial_stage_state() -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "round": 0,
            "children": {},
        }
        for stage in STAGE_ORDER
    }


def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def apply_event_to_stage_state(
    stage_state: dict[str, dict[str, Any]],
    ev: PipelineEvent,
    now: float,
) -> None:
    if ev.kind == "round_advanced":
        stage_state["Investment Debate"]["round"] = ev.round or 0
        return
    if ev.kind in ("completed", "errored", "values"):
        return
    if ev.node_name is None or ev.node_name not in STAGE_FOR_NODE:
        return

    stage = STAGE_FOR_NODE[ev.node_name]
    info = stage_state[stage]
    children = info["children"]
    child = children.setdefault(
        ev.node_name,
        {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "size_pct": None,
        },
    )

    if ev.status == "started":
        child["status"] = "running"
        child["started_at"] = now
        if info["status"] == "pending":
            info["status"] = "running"
            info["started_at"] = now
    elif ev.status == "completed":
        child["status"] = "completed"
        child["completed_at"] = now
        if ev.position_size_pct is not None:
            pct = ev.position_size_pct
            child["size_pct"] = pct * 100 if pct < 1 else pct
        if children and all(
            c["status"] in ("completed", "errored", "cancelled")
            for c in children.values()
        ):
            info["status"] = "completed"
            info["completed_at"] = now
    elif ev.status == "errored":
        child["status"] = "errored"
        child["completed_at"] = now
        info["status"] = "errored"
        info["completed_at"] = now
    elif ev.status == "cancelled":
        child["status"] = "cancelled"
        child["completed_at"] = now


def drain_queue_and_apply(
    stage_state: dict[str, dict[str, Any]],
    q: "queue.Queue[Any]",
) -> tuple[bool, Optional[PipelineEvent]]:
    last_event: Optional[PipelineEvent] = None
    now = time.monotonic()
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break
        apply_event_to_stage_state(stage_state, ev, now)
        last_event = ev
        if ev.kind in ("completed", "errored"):
            return False, ev
    return True, last_event


def drain_to_status(
    stage_state: dict[str, dict[str, Any]],
    q: "queue.Queue[Any]",
) -> tuple[bool, Optional[PipelineEvent]]:
    return drain_queue_and_apply(stage_state, q)


def _run_in_background(
    graph: Any,
    initial_state: Any,
    config: dict[str, Any],
    q: "queue.Queue[Any]",
    cancel_event: threading.Event,
) -> None:
    from financialagents.observability.events import stream_pipeline_events

    async def _drain() -> None:
        async for ev in stream_pipeline_events(
            graph, initial_state, config, cancel_event=cancel_event
        ):
            q.put(ev)

    try:
        asyncio.run(_drain())
    except BaseException as exc:  # pragma: no cover - exceptional path
        q.put(PipelineEvent(kind="errored", status="errored", error=exc))


def start_pipeline(
    graph: Any,
    initial_state: Any,
    config: dict[str, Any],
) -> None:
    q: queue.Queue[Any] = queue.Queue()
    cancel_event = threading.Event()
    st.session_state.pipeline_queue = q
    st.session_state.pipeline_running = True
    st.session_state.pipeline_cancel = cancel_event
    st.session_state.stage_state = _initial_stage_state()
    st.session_state.pipeline_started_at = time.monotonic()
    st.session_state.last_completed_event = None
    t = threading.Thread(
        target=_run_in_background,
        args=(graph, initial_state, config, q, cancel_event),
        daemon=True,
    )
    st.session_state.pipeline_thread = t
    t.start()


def cancel_pipeline() -> None:
    cancel_event = st.session_state.get("pipeline_cancel")
    thread = st.session_state.get("pipeline_thread")
    if cancel_event is not None:
        try:
            cancel_event.set()
        except Exception:
            pass
    st.session_state.pipeline_running = False
    if thread is not None:
        try:
            thread.join(timeout=0.5)
        except Exception:
            pass
    if "pipeline_thread" in st.session_state:
        st.session_state.pipeline_thread = None
    if "pipeline_cancel" in st.session_state:
        st.session_state.pipeline_cancel = None


def _streamlit_state_for_stage(
    status_label: str,
) -> Literal["running", "complete", "error"]:
    if status_label == "completed":
        return "complete"
    if status_label == "errored":
        return "error"
    return "running"


def render_live_region() -> None:
    stage_state = st.session_state.get("stage_state") or _initial_stage_state()
    for stage in STAGE_ORDER:
        info = stage_state[stage]
        status_label = info["status"]
        elapsed: Optional[float] = None
        if info["started_at"]:
            elapsed = (info["completed_at"] or time.monotonic()) - info["started_at"]
        label = f"{stage} · {_fmt_elapsed(elapsed)}".rstrip(" ·")
        if stage == "Investment Debate" and info["round"]:
            label += f" · round {info['round']}"
        with st.status(
            label,
            state=_streamlit_state_for_stage(status_label),
            expanded=(status_label != "completed"),
        ):
            for node_name, child in info["children"].items():
                suffix = ""
                if child.get("size_pct") is not None:
                    suffix = f" · {child['size_pct']:.1f}%"
                icon = STATUS_ICON.get(child["status"], STATUS_ICON["pending"])
                st.write(f"{icon} {node_name}{suffix}")


@st.fragment(run_every=0.5)
def live_fragment() -> None:
    if not st.session_state.get("pipeline_running"):
        return
    q = st.session_state.get("pipeline_queue")
    if q is None:
        return
    stage_state = st.session_state.get("stage_state") or _initial_stage_state()
    still_running, last_event = drain_queue_and_apply(stage_state, q)
    st.session_state.stage_state = stage_state
    if not still_running:
        st.session_state.pipeline_running = False
        st.session_state.last_completed_event = last_event
        st.rerun()


__all__ = [
    "STAGE_ORDER",
    "STATUS_ICON",
    "apply_event_to_stage_state",
    "cancel_pipeline",
    "drain_queue_and_apply",
    "drain_to_status",
    "live_fragment",
    "render_live_region",
    "start_pipeline",
]
