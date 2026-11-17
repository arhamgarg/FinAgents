from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import AsyncIterator
from types import FrameType
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from finagents.clock import local_now
from finagents.observability.events import STAGE_FOR_NODE, PipelineEvent

SECONDS_PER_MINUTE = 60

STAGE_ORDER: list[str] = [
    "Analysts",
    "Research Manager",
    "Investment Debate",
    "Trader",
    "Risk Debate + PM",
]

PERSONA_TEMP: dict[str, float] = {
    "aggressive_risk_node": 0.9,
    "neutral_risk_node": 0.6,
    "conservative_risk_node": 0.3,
}


def _initial_stage_state(max_debate_rounds: int) -> dict[str, dict]:
    return {
        stage: {
            "started_at": None,
            "completed_at": None,
            "round": 0,
            "max_rounds": max_debate_rounds,
            "children": {},
        }
        for stage in STAGE_ORDER
    }


def _fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), SECONDS_PER_MINUTE)
    h, m = divmod(m, SECONDS_PER_MINUTE)
    return f"{h}:{m:02d}:{s:02d}"


def _glyph_for(status: str) -> Spinner | Text:
    if status == "running":
        return Spinner("dots", style="cyan")
    if status == "completed":
        return Text("✓", style="green")
    if status == "errored":
        return Text("✗", style="red")
    if status == "cancelled":
        return Text("⛯", style="yellow")
    return Text(" ", style="dim")


def _build_header(
    ticker: str,
    date: str,
    provider: str,
    model: str,
    total_elapsed: float,
) -> Text:
    t = Text()
    t.append(ticker, style="bold")
    t.append(f" · {date} · {provider}:{model} · {_fmt_elapsed(total_elapsed)}")
    return t


def make_stage_table(  # noqa: PLR0915
    stage_state: dict[str, dict],
    header: Text,
    now: float,
) -> Table:
    outer = Table.grid(padding=(0, 0))
    outer.add_column(overflow="ellipsis")
    outer.add_row(header)
    outer.add_row(Text(""))

    for stage in STAGE_ORDER:
        info = stage_state[stage]
        stage_started = info["started_at"]
        stage_done = info["completed_at"]
        stage_elapsed: float | None = None
        if stage_started:
            stage_elapsed = (stage_done or now) - stage_started

        label = Text(stage, style="bold")
        if stage == "Investment Debate" and info["round"]:
            label.append(
                f" · round {info['round']}/{info['max_rounds']}",
                style="dim",
            )

        stage_row = Table.grid(padding=(0, 1), expand=False)
        stage_row.add_column(justify="left", overflow="ellipsis")
        stage_row.add_column(justify="right", overflow="crop")
        stage_row.add_row(label, Text(_fmt_elapsed(stage_elapsed), style="dim"))
        outer.add_row(stage_row)

        for node_name, child in info["children"].items():
            child_started = child["started_at"]
            child_done = child["completed_at"]
            child_status = child["status"]
            child_elapsed: float | None = None
            if child_started:
                child_elapsed = (child_done or now) - child_started

            left = Text("  ")
            left.append(node_name)
            if node_name in PERSONA_TEMP:
                left.append(f" · T={PERSONA_TEMP[node_name]}", style="dim")
            if child.get("size_pct") is not None:
                left.append(f" · {child['size_pct']:.1f}%", style="dim")

            glyph = _glyph_for(child_status)

            child_row = Table.grid(padding=(0, 1), expand=False)
            child_row.add_column(justify="left", overflow="ellipsis")
            child_row.add_column(justify="right", overflow="crop")

            if isinstance(glyph, Spinner):
                composite = Table.grid(padding=(0, 1), expand=False)
                composite.add_column()
                composite.add_column(overflow="ellipsis")
                composite.add_row(glyph, left)
                child_row.add_row(composite, Text(_fmt_elapsed(child_elapsed), style="dim"))
            else:
                static_left = Text()
                static_left.append("  ")
                if isinstance(glyph, Text):
                    static_left.append_text(glyph)
                    static_left.append(" ")
                static_left.append(node_name)
                if node_name in PERSONA_TEMP:
                    static_left.append(f" · T={PERSONA_TEMP[node_name]}", style="dim")
                if child.get("size_pct") is not None:
                    static_left.append(f" · {child['size_pct']:.1f}%", style="dim")
                child_row.add_row(
                    static_left,
                    Text(_fmt_elapsed(child_elapsed), style="dim"),
                )
            outer.add_row(child_row)

    return outer


def _apply_event(
    stage_state: dict[str, dict],
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
        if info["started_at"] is None:
            info["started_at"] = now
    elif ev.status == "completed":
        child["status"] = "completed"
        child["completed_at"] = now
        if ev.position_size_pct is not None:
            pct = ev.position_size_pct
            child["size_pct"] = pct * 100 if pct < 1 else pct
        if all(c["status"] in ("completed", "errored", "cancelled") for c in children.values()):
            info["completed_at"] = now
    elif ev.status == "errored":
        child["status"] = "errored"
        child["completed_at"] = now
    elif ev.status == "cancelled":
        child["status"] = "cancelled"
        child["completed_at"] = now


async def run_with_log_lines(  # noqa: PLR0912
    events: AsyncIterator[PipelineEvent],
    *,
    ticker: str,
    date: str,
    provider: str,
    model: str,
    log_console: Console,
    cancel_flag: asyncio.Event | None = None,
) -> Any | None:
    log_console.print(f"# {ticker} · {date} · {provider}:{model}", style="dim")
    final_state: Any | None = None
    last_values_snapshot: Any | None = None
    errored_exc: BaseException | None = None

    async for ev in events:
        ts = local_now().strftime("%H:%M:%S")
        if ev.kind == "values" and ev.final_state is not None:
            last_values_snapshot = ev.final_state

        if ev.kind == "node_status":
            if ev.status == "started":
                log_console.print(f"[{ts}] {ev.node_name} started")
            elif ev.status == "completed":
                extra = ""
                if ev.position_size_pct is not None:
                    pct = (
                        ev.position_size_pct * 100
                        if ev.position_size_pct < 1
                        else ev.position_size_pct
                    )
                    extra = f" {pct:.1f}%"
                log_console.print(f"[{ts}] {ev.node_name} done{extra}")
            elif ev.status == "errored":
                log_console.print(f"[{ts}] {ev.node_name} errored", style="red")
            elif ev.status == "cancelled":
                log_console.print(f"[{ts}] {ev.node_name} cancelled", style="yellow")
        elif ev.kind == "completed":
            final_state = ev.final_state
            break
        elif ev.kind == "errored":
            if ev.error is not None:
                errored_exc = ev.error
            break

        if (
            cancel_flag is not None
            and cancel_flag.is_set()
            and ev.kind == "node_status"
            and ev.status in ("completed", "errored", "cancelled")
        ):
            break

    if errored_exc is not None:
        raise errored_exc

    return final_state if final_state is not None else last_values_snapshot


async def run_with_live_render(
    events: AsyncIterator[PipelineEvent],
    *,
    ticker: str,
    date: str,
    provider: str,
    model: str,
    max_debate_rounds: int = 3,
    console: Console | None = None,
    cancel_flag: asyncio.Event | None = None,
) -> Any | None:
    if console is None:
        console = Console()

    if not console.is_terminal:
        return await run_with_log_lines(
            events,
            ticker=ticker,
            date=date,
            provider=provider,
            model=model,
            log_console=console,
            cancel_flag=cancel_flag,
        )

    stage_state = _initial_stage_state(max_debate_rounds)
    start = time.monotonic()
    header = _build_header(ticker, date, provider, model, 0.0)

    final_state: Any | None = None
    last_values_snapshot: Any | None = None
    errored_exc: BaseException | None = None

    with Live(
        make_stage_table(stage_state, header, start),
        refresh_per_second=8,
        vertical_overflow="ellipsis",
        console=console,
        transient=False,
    ) as live:
        async for ev in events:
            now = time.monotonic()

            if ev.kind == "values" and ev.final_state is not None:
                last_values_snapshot = ev.final_state

            _apply_event(stage_state, ev, now)
            header = _build_header(ticker, date, provider, model, now - start)
            live.update(make_stage_table(stage_state, header, now))

            if (
                cancel_flag is not None
                and cancel_flag.is_set()
                and ev.kind == "node_status"
                and ev.status in ("completed", "errored", "cancelled")
            ):
                break

            if ev.kind == "completed":
                final_state = ev.final_state
                break
            if ev.kind == "errored":
                if ev.error is not None:
                    errored_exc = ev.error
                break

    if errored_exc is not None:
        raise errored_exc

    return final_state if final_state is not None else last_values_snapshot


def install_sigint_handler() -> asyncio.Event:
    cancel_flag = asyncio.Event()
    sigint_count = {"n": 0}

    def _handler(signum: int, frame: FrameType | None) -> None:
        sigint_count["n"] += 1
        if sigint_count["n"] == 1:
            cancel_flag.set()
        else:
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    return cancel_flag


__all__ = [
    "PERSONA_TEMP",
    "STAGE_ORDER",
    "install_sigint_handler",
    "make_stage_table",
    "run_with_live_render",
    "run_with_log_lines",
]
