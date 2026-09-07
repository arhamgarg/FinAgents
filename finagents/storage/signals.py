from __future__ import annotations

import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import aiosqlite

Status = Literal["running", "success", "cancelled", "errored"]
_VALID_STATUSES = frozenset({"running", "success", "cancelled", "errored"})
_RUN_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _check_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(f"Unknown signal status: {status!r}")


def _check_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id (not 8-hex): {run_id!r}")


@asynccontextmanager
async def open_signals_db(
    db_path: Path,
) -> AsyncIterator[aiosqlite.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
        yield db


async def insert_running(
    run_id: str,
    ticker: str,
    date: str,
    *,
    provider: str,
    model: str,
    db_path: Path,
) -> None:
    _check_run_id(run_id)
    async with open_signals_db(db_path) as db:
        await db.execute(
            "INSERT INTO signals (run_id, ticker, date, created_at, "
            "provider, model, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
            (
                run_id,
                ticker,
                date,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                provider,
                model,
            ),
        )
        await db.commit()


async def update_success(
    run_id: str,
    *,
    action: str,
    confidence: float,
    position_size_pct: float,
    approved: bool,
    pm_decision: str,
    payload_json: str,
    db_path: Path,
) -> None:
    _check_run_id(run_id)
    async with open_signals_db(db_path) as db:
        await db.execute(
            "UPDATE signals SET status='success', action=?, confidence=?, "
            "position_size_pct=?, approved=?, pm_decision=?, payload_json=? "
            "WHERE run_id=?",
            (
                action,
                confidence,
                position_size_pct,
                1 if approved else 0,
                pm_decision,
                payload_json,
                run_id,
            ),
        )
        await db.commit()


async def update_cancelled(
    run_id: str,
    *,
    error_msg: str,
    payload_json: Optional[str] = None,
    db_path: Path,
) -> None:
    _check_run_id(run_id)
    async with open_signals_db(db_path) as db:
        await db.execute(
            "UPDATE signals SET status='cancelled', error_msg=?, "
            "payload_json=COALESCE(?, payload_json) WHERE run_id=?",
            (error_msg, payload_json, run_id),
        )
        await db.commit()


async def update_errored(
    run_id: str,
    *,
    error_msg: str,
    payload_json: Optional[str] = None,
    db_path: Path,
) -> None:
    _check_run_id(run_id)
    async with open_signals_db(db_path) as db:
        await db.execute(
            "UPDATE signals SET status='errored', error_msg=?, "
            "payload_json=COALESCE(?, payload_json) WHERE run_id=?",
            (error_msg, payload_json, run_id),
        )
        await db.commit()


async def load_history(
    db_path: Path,
    *,
    limit: int = 20,
    ticker: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    include_payload: bool = True,
) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    where_clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        where_clauses.append("ticker = ?")
        params.append(ticker)
    if since:
        where_clauses.append("created_at >= ?")
        params.append(since)
    if until:
        where_clauses.append("created_at < ?")
        params.append(until)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    payload_column = ", payload_json " if include_payload else " "
    sql = (
        "SELECT run_id, ticker, date, created_at, provider, model, status, "
        "action, confidence, position_size_pct, approved, pm_decision, "
        "error_msg" + payload_column + f"FROM signals {where} "
        "ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    async with open_signals_db(db_path) as db:
        async with db.execute(sql, tuple(params)) as cur:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def load_signal_by_run_id(
    db_path: Path,
    run_id: str,
) -> Optional[dict[str, Any]]:
    try:
        _check_run_id(run_id)
    except ValueError:
        return None
    if not db_path.exists():
        return None
    async with open_signals_db(db_path) as db:
        async with db.execute(
            "SELECT run_id, ticker, date, created_at, provider, model, "
            "status, action, confidence, position_size_pct, approved, "
            "pm_decision, error_msg, payload_json "
            "FROM signals WHERE run_id=?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


async def prune_before(db_path: Path, before_date: str) -> int:
    if not db_path.exists():
        return 0
    async with open_signals_db(db_path) as db:
        cur = await db.execute(
            "DELETE FROM signals WHERE created_at < ?",
            (before_date,),
        )
        await db.commit()
        return cur.rowcount or 0
