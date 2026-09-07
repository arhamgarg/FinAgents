from __future__ import annotations

from typing import Optional

from rich.console import Console

from finagents.config import Settings
from finagents.storage.paths import signals_db_path
from finagents.storage.signals import load_signal_by_run_id

NOT_FOUND_TEMPLATE = (
    "No signal found with run_id {run_id}. Try `fa history` to list available runs."
)


async def run_show(
    run_id: str,
    console: Optional[Console] = None,
) -> int:
    if console is None:
        console = Console()
    db_path = signals_db_path(Settings())

    row = await load_signal_by_run_id(db_path, run_id)
    if row is None:
        console.print(
            NOT_FOUND_TEMPLATE.format(run_id=run_id),
            style="yellow",
        )
        return 1

    payload = row.get("payload_json")
    if not payload:
        status = row.get("status") or "unknown"
        console.print(
            f"Signal {run_id} has status='{status}' with no full payload "
            "(run was cancelled or errored before completion).",
            style="yellow",
        )
        return 0

    from pydantic import ValidationError

    from finagents.agents.utils.agent_states import AgentState
    from finagents.cli.panels import render_signal_card

    try:
        state = AgentState.model_validate_json(payload)
    except (ValueError, ValidationError) as exc:
        console.print(f"[red]Failed to deserialize payload_json: {exc}[/red]")
        return 1

    if state.final_trade_decision is None:
        console.print(
            f"Signal {run_id} payload has no final_trade_decision; "
            "the run did not reach the portfolio manager.",
            style="yellow",
        )
        return 0

    console.print(render_signal_card(state))
    return 0
