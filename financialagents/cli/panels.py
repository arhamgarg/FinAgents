from __future__ import annotations

from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

VERDICT_STYLE: dict[str, str] = {
    "BUY": "bold green",
    "SELL": "bold red",
    "HOLD": "bold yellow",
}

APPROVAL_STYLE: dict[str, str] = {
    "APPROVED": "green",
    "MODIFIED": "yellow",
    "VETOED": "red",
}


def _approval_label(decision: str, approved: bool) -> str:
    if decision == "veto":
        return "VETOED"
    if decision == "modify":
        return "MODIFIED"
    return "APPROVED" if approved else "VETOED"


def _format_inr(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.0f}"


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def render_signal_card(state) -> Panel:
    pd_ = state.final_trade_decision
    ts = pd_.signal
    action = ts.action
    approval = _approval_label(pd_.decision, pd_.approved)
    title = f"{state.ticker} · {state.date}"

    body = Table.grid(padding=(0, 1))
    body.add_column()

    verdict_line = Text()
    verdict_line.append(action, style=VERDICT_STYLE.get(action, "bold"))
    verdict_line.append(" ")
    verdict_line.append(f"[{approval}]", style=APPROVAL_STYLE[approval])
    body.add_row(verdict_line)
    body.add_row(Text(""))

    metric_row = Table.grid(padding=(0, 2))
    for _ in range(4):
        metric_row.add_column(justify="left")
    metric_row.add_row("Entry", "Stop", "Target", "Size")
    metric_row.add_row(
        _format_inr(ts.entry_price),
        _format_inr(ts.stop_loss),
        _format_inr(ts.target_1),
        _format_pct(ts.position_size_pct),
    )
    body.add_row(metric_row)
    body.add_row(Text(""))

    if pd_.pm_notes:
        pm_line = Text("pm: ", style="dim")
        pm_line.append(pd_.pm_notes)
        body.add_row(pm_line)

    return Panel(
        body,
        title=title,
        title_align="left",
        border_style="bright_black",
    )


def render_error_panel(
    error: BaseException,
    hint: Optional[str] = None,
) -> Panel:
    body = Text(str(error), style="red")
    if hint:
        body.append("\n")
        body.append(hint, style="dim")
    return Panel(
        body,
        title="Error",
        title_align="left",
        border_style="red",
    )


def render_cancelled_panel(node_name: str, run_id: str) -> Panel:
    body = Text(
        f"Pipeline stopped after {node_name}. "
        f"Partial state saved to signal log (run_id {run_id}).",
        style="yellow",
    )
    return Panel(
        body,
        title="Cancelled",
        title_align="left",
        border_style="yellow",
    )
