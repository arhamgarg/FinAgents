from __future__ import annotations

import csv as _csv
import json
import sys
from pathlib import Path
from typing import IO, Any

from rich.console import Console
from rich.prompt import Confirm

from finagents.config import Settings
from finagents.storage.paths import signals_db_path
from finagents.storage.signals import load_history

EXPORT_COLS: tuple[str, ...] = (
    "run_id",
    "ticker",
    "date",
    "created_at",
    "provider",
    "model",
    "status",
    "action",
    "confidence",
    "position_size_pct",
    "approved",
    "pm_decision",
    "error_msg",
)

NO_ROWS_COPY = "No signals match the given filters. Showing 0 rows."
NON_TTY_OVERWRITE_REFUSAL = (
    "Refusing to overwrite existing file in non-interactive mode without --yes."
)


def _slim_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in EXPORT_COLS:
        val = row.get(col)
        if col == "approved" and val is not None:
            val = bool(val)
        out[col] = val
    return out


async def run_export(  # noqa: PLR0912
    *,
    format_: str,
    ticker: str | None,
    since: str | None,
    until: str | None,
    out: str | None,
    yes: bool = False,
) -> int:
    if format_ not in ("json", "csv"):
        sys.stderr.write(f"Unsupported format: {format_}. Use json or csv.\n")
        return 1

    db_path = signals_db_path(Settings())
    rows = await load_history(
        db_path,
        limit=10_000,
        ticker=ticker,
        since=since,
        until=until,
        include_payload=False,
    )

    stream: IO[str]
    target: Path | None = None
    if out is not None:
        target = Path(out)
        if target.exists() and not yes:
            stderr_console = Console(file=sys.stderr)
            if not sys.stdin.isatty():
                stderr_console.print(NON_TTY_OVERWRITE_REFUSAL, style="red")
                return 1
            ok = Confirm.ask(
                f"{target} exists. Overwrite?",
                default=False,
                console=stderr_console,
            )
            if not ok:
                stderr_console.print("[dim]Cancelled.[/dim]")
                return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = target.open("w", encoding="utf-8", newline="")
    else:
        stream = sys.stdout

    try:
        if not rows:
            Console(file=sys.stderr).print(NO_ROWS_COPY)
            if format_ == "json":
                stream.write("[]\n")
            else:
                stream.write(",".join(EXPORT_COLS) + "\n")
            return 0

        if format_ == "json":
            slim = [_slim_row(r) for r in rows]
            json.dump(slim, stream, indent=2, default=str)
            stream.write("\n")
        else:
            writer = _csv.DictWriter(stream, fieldnames=list(EXPORT_COLS))
            writer.writeheader()
            for r in rows:
                slim = _slim_row(r)
                writer.writerow({k: ("" if v is None else v) for k, v in slim.items()})
    finally:
        if target is not None:
            stream.close()

    return 0
