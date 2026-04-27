from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from financialagents.config import Settings
from financialagents.storage.paths import signals_db_path
from financialagents.storage.signals import (
    load_history,
    prune_before,
)

EMPTY_HISTORY_COPY = (
    "No signals yet. Run `fa run <ticker>` to generate your first signal."
)

NON_TTY_PRUNE_REFUSAL = (
    "Refusing to prune in non-interactive mode without --yes."
)

DEFAULT_COLS: tuple[str, ...] = (
    "created_at",
    "ticker",
    "action",
    "confidence",
    "position_size_pct",
    "approved",
    "status",
)


async def run_history(
    *,
    ticker: Optional[str],
    since: Optional[str],
    limit: int,
    show_all: bool,
    expand: bool,
    prune: bool,
    before: Optional[str],
    yes: bool = False,
    console: Optional[Console] = None,
) -> int:
    if console is None:
        console = Console()
    db_path = signals_db_path(Settings())

    if prune:
        return await _run_prune(db_path, before, yes, console)

    eff_limit = 10_000 if show_all else limit
    rows = await load_history(
        db_path,
        limit=eff_limit,
        ticker=ticker,
        since=since,
    )
    if not rows:
        console.print(EMPTY_HISTORY_COPY)
        return 0

    table = Table(show_header=True, header_style="bold")
    for col in DEFAULT_COLS:
        table.add_column(col)
    for row in rows:
        table.add_row(
            _short_ts(row.get("created_at")),
            str(row.get("ticker") or "—"),
            _color_action(row.get("action")),
            _fmt_float(row.get("confidence"), "{:.2f}"),
            _fmt_pct(row.get("position_size_pct")),
            "✓" if row.get("approved") else "—",
            _color_status(row.get("status") or "—"),
        )
    console.print(table)

    if expand:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.cli.panels import render_signal_card

        for row in rows:
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                state = AgentState.model_validate_json(payload)
            except Exception:
                continue
            if state.final_trade_decision is None:
                continue
            console.print(render_signal_card(state))
    return 0


async def _run_prune(
    db_path: Path,
    before: Optional[str],
    yes: bool,
    console: Console,
) -> int:
    if before is None:
        console.print("[red]--prune requires --before <date>.[/red]")
        return 1

    rows = await load_history(db_path, limit=10_000)
    n_match = sum(1 for r in rows if (r.get("created_at") or "") < before)

    is_tty = sys.stdin.isatty()
    if not is_tty and not yes:
        console.print(NON_TTY_PRUNE_REFUSAL, style="red")
        return 1

    if not yes:
        ok = Confirm.ask(
            f"Delete {n_match} signals dated before {before}? "
            "This cannot be undone.",
            default=False,
            console=console,
        )
        if not ok:
            console.print("[dim]Cancelled.[/dim]")
            return 0

    deleted = await prune_before(db_path, before)
    console.print(f"Deleted {deleted} signals.")
    return 0


def _short_ts(value: Any) -> str:
    if not value:
        return "—"
    s = str(value)
    return s[:16].replace("T", " ")


def _fmt_float(value: Any, fmt: str) -> str:
    if value is None:
        return "—"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v < 1:
        return f"{v * 100:.1f}%"
    return f"{v:.1f}%"


def _color_action(action: Any) -> str:
    if action == "BUY":
        return "[green]BUY[/green]"
    if action == "SELL":
        return "[red]SELL[/red]"
    if action == "HOLD":
        return "[yellow]HOLD[/yellow]"
    return "—"


def _color_status(status: Any) -> str:
    color_map = {
        "success": "green",
        "cancelled": "yellow",
        "errored": "red",
        "running": "dim",
    }
    s = str(status)
    color = color_map.get(s, "dim")
    return f"[{color}]{s}[/{color}]"
