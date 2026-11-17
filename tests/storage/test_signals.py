from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from finagents.agents.utils.agent_states import AgentState
from finagents.storage.migrations import run_migrations
from finagents.storage.signals import (
    insert_running,
    load_history,
    load_signal_by_run_id,
    open_signals_db,
    prune_before,
    update_cancelled,
    update_errored,
    update_success,
)


class TestSignalsSchema:
    @pytest.mark.asyncio
    async def test_first_open_creates_schema(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        conn = sqlite3.connect(tmp_signals_db)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            mode = conn.execute("PRAGMA journal_mode").fetchone()
        finally:
            conn.close()
        assert "signals" in tables
        assert "_meta" in tables
        assert mode is not None and mode[0].lower() == "wal"


class TestSignalsCRUD:
    @pytest.mark.asyncio
    async def test_insert_at_start(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "abc12345",
            "RELIANCE",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        conn = sqlite3.connect(tmp_signals_db)
        try:
            rows = conn.execute(
                "SELECT run_id, ticker, date, created_at, provider, model, "
                "status, action, confidence, position_size_pct, approved, "
                "pm_decision, error_msg, payload_json FROM signals"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "abc12345"
        assert row[1] == "RELIANCE"
        assert row[2] == "2026-04-26"
        assert row[3] is not None and len(row[3]) > 0
        assert row[4] == "openrouter"
        assert row[5] == "openrouter/auto"
        assert row[6] == "running"
        assert row[7] is None
        assert row[8] is None
        assert row[9] is None
        assert row[10] is None
        assert row[11] is None
        assert row[12] is None
        assert row[13] is None

    @pytest.mark.asyncio
    async def test_update_on_end(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "abc12345",
            "RELIANCE",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        state = AgentState(ticker="RELIANCE", date="2026-04-26")
        payload = state.model_dump_json(exclude={"messages"})
        await update_success(
            "abc12345",
            action="BUY",
            confidence=0.72,
            position_size_pct=3.0,
            approved=True,
            pm_decision="approved",
            payload_json=payload,
            db_path=tmp_signals_db,
        )
        loaded = await load_signal_by_run_id(tmp_signals_db, "abc12345")
        assert loaded is not None
        assert loaded["status"] == "success"
        assert loaded["action"] == "BUY"
        assert loaded["confidence"] == pytest.approx(0.72)
        assert loaded["position_size_pct"] == pytest.approx(3.0)
        assert loaded["approved"] == 1
        assert loaded["pm_decision"] == "approved"
        assert loaded["payload_json"] is not None
        reloaded = AgentState.model_validate_json(loaded["payload_json"])
        assert reloaded.ticker == "RELIANCE"
        assert reloaded.date == "2026-04-26"

    @pytest.mark.asyncio
    async def test_partial_persist(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "abc12345",
            "RELIANCE",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        await update_cancelled(
            "abc12345",
            error_msg="user",
            db_path=tmp_signals_db,
        )
        loaded = await load_signal_by_run_id(tmp_signals_db, "abc12345")
        assert loaded is not None
        assert loaded["status"] == "cancelled"
        assert loaded["error_msg"] == "user"

    @pytest.mark.asyncio
    async def test_update_errored_writes_status_and_error_msg(
        self,
        tmp_signals_db: Path,
    ) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "deadbeef",
            "TCS",
            "2026-04-26",
            provider="groq",
            model="openai/gpt-oss-120b",
            db_path=tmp_signals_db,
        )
        await update_errored(
            "deadbeef",
            error_msg="upstream timeout",
            db_path=tmp_signals_db,
        )
        loaded = await load_signal_by_run_id(tmp_signals_db, "deadbeef")
        assert loaded is not None
        assert loaded["status"] == "errored"
        assert loaded["error_msg"] == "upstream timeout"


class TestSignalsConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "abc12345",
            "RELIANCE",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )

        async def writer() -> None:
            await insert_running(
                "deadbeef",
                "TCS",
                "2026-04-26",
                provider="openrouter",
                model="openrouter/auto",
                db_path=tmp_signals_db,
            )

        async def reader() -> list[dict]:
            return await load_history(tmp_signals_db, limit=10)

        results = await asyncio.wait_for(
            asyncio.gather(writer(), reader()),
            timeout=5.0,
        )
        history = results[1]
        assert isinstance(history, list)
        assert any(r["run_id"] == "abc12345" for r in history)

        full = await load_history(tmp_signals_db, limit=10)
        run_ids = {r["run_id"] for r in full}
        assert run_ids == {"abc12345", "deadbeef"}


class TestSignalsPayloadRoundTrip:
    @pytest.mark.asyncio
    async def test_payload_json_roundtrip(self, tmp_signals_db: Path) -> None:
        await run_migrations(tmp_signals_db)
        state = AgentState(ticker="INFY", date="2026-04-26")
        payload = state.model_dump_json(exclude={"messages"})

        await insert_running(
            "0badc0de",
            "INFY",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        await update_success(
            "0badc0de",
            action="HOLD",
            confidence=0.5,
            position_size_pct=0.0,
            approved=False,
            pm_decision="vetoed",
            payload_json=payload,
            db_path=tmp_signals_db,
        )
        loaded = await load_signal_by_run_id(tmp_signals_db, "0badc0de")
        assert loaded is not None
        reloaded = AgentState.model_validate_json(loaded["payload_json"])
        assert reloaded.ticker == state.ticker
        assert reloaded.date == state.date


class TestSignalsPragmas:
    @pytest.mark.asyncio
    async def test_synchronous_normal_per_connection(
        self,
        tmp_signals_db: Path,
    ) -> None:
        await run_migrations(tmp_signals_db)
        async with (
            open_signals_db(tmp_signals_db) as db,
            db.execute("PRAGMA synchronous") as cur,
        ):
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 1


class TestSignalsRunIdValidation:
    @pytest.mark.asyncio
    async def test_load_with_invalid_run_id_returns_none(
        self,
        tmp_signals_db: Path,
    ) -> None:
        await run_migrations(tmp_signals_db)
        result = await load_signal_by_run_id(
            tmp_signals_db,
            "../../etc/passwd",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_insert_with_invalid_run_id_raises(
        self,
        tmp_signals_db: Path,
    ) -> None:
        await run_migrations(tmp_signals_db)
        with pytest.raises(ValueError, match="run_id"):
            await insert_running(
                "notHEX!!",
                "RELIANCE",
                "2026-04-26",
                provider="openrouter",
                model="openrouter/auto",
                db_path=tmp_signals_db,
            )


class TestSignalsPrune:
    @pytest.mark.asyncio
    async def test_prune_before_returns_zero_on_missing_db(
        self,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "does_not_exist.db"
        assert not missing.exists(), "fixture sanity: target path must not exist"
        n = await prune_before(missing, "2026-01-01")
        assert n == 0
        assert not missing.exists(), "prune_before must not create the DB file as a side-effect"

    @pytest.mark.asyncio
    async def test_prune_before_returns_count_after_migrations(
        self,
        tmp_signals_db: Path,
    ) -> None:
        await run_migrations(tmp_signals_db)
        await insert_running(
            "aaaaaaaa",
            "RELIANCE",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        await insert_running(
            "bbbbbbbb",
            "TCS",
            "2026-04-26",
            provider="openrouter",
            model="openrouter/auto",
            db_path=tmp_signals_db,
        )
        n = await prune_before(tmp_signals_db, "2099-01-01")
        assert n == 2
        remaining = await load_history(tmp_signals_db, limit=10)
        assert remaining == []


_ = aiosqlite


async def test_summary_projection_omits_payload(migrated_db):
    from finagents.storage.signals import insert_running, load_history

    await insert_running(
        "aabbccdd",
        "RELIANCE",
        "2024-12-31",
        provider="fixture",
        model="fixture",
        db_path=migrated_db,
    )
    full = await load_history(migrated_db)
    summary = await load_history(migrated_db, include_payload=False)
    assert "payload_json" in full[0]
    assert "payload_json" not in summary[0]
    assert summary[0] == {k: v for k, v in full[0].items() if k != "payload_json"}
