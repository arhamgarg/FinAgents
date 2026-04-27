from __future__ import annotations

import asyncio
import io
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console

from financialagents.agents.managers.schemas import PortfolioDecision
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.agents.utils.agent_states import AgentState
from financialagents.storage.migrations import run_migrations
from financialagents.storage.signals import insert_running, update_success


def _seed_one_row(db_path: Path, run_id: str = "abc12345") -> AgentState:
    asyncio.run(run_migrations(db_path))

    ts = TradingSignal(
        action="BUY",
        conviction=0.72,
        entry_price=1234.0,
        stop_loss=1180.0,
        target_1=1310.0,
        target_2=1380.0,
        position_size_pct=0.03,
        position_size_shares=10,
        risk_reward_ratio=1.45,
        rationale="strong technicals",
        analyst_weight_applied={
            "technical": 0.303,
            "fundamentals": 0.455,
            "news": 0.242,
        },
        sizing_method="atr_based",
        time_horizon="positional",
        t2t_flag=False,
    )
    pd_ = PortfolioDecision(
        decision="approve",
        approved=True,
        pm_notes="ok",
        signal=ts,
    )
    state = AgentState(
        ticker="RELIANCE",
        date="2026-04-26",
        final_trade_decision=pd_,
    )
    payload = state.model_dump_json(exclude={"messages"})
    asyncio.run(
        insert_running(
            run_id, "RELIANCE", "2026-04-26",
            provider="ollama", model="llama3.1:8b",
            db_path=db_path,
        )
    )
    asyncio.run(
        update_success(
            run_id,
            action="BUY",
            confidence=0.72,
            position_size_pct=0.03,
            approved=True,
            pm_decision="approve",
            payload_json=payload,
            db_path=db_path,
        )
    )
    return state


def _make_recording_console() -> Console:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200, record=True)


@pytest.fixture
def patched_db(monkeypatch, tmp_signals_db):
    monkeypatch.setattr(
        "financialagents.cli.show.signals_db_path",
        lambda settings=None: tmp_signals_db,
    )
    return tmp_signals_db


class TestFaShow:
    def test_show_renders_panel(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.show import run_show

        _seed_one_row(tmp_signals_db, run_id="abc12345")
        console = _make_recording_console()
        rc = asyncio.run(run_show("abc12345", console=console))
        assert rc == 0
        out = console.export_text()
        assert "BUY" in out
        assert "RELIANCE" in out
        assert "2026-04-26" in out

    def test_show_not_found(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.show import run_show

        _seed_one_row(tmp_signals_db, run_id="abc12345")
        console = _make_recording_console()
        rc = asyncio.run(run_show("00000000", console=console))
        assert rc == 1
        out = console.export_text()
        assert (
            "No signal found with run_id 00000000. "
            "Try `fa history` to list available runs."
        ) in out

    def test_show_invalid_run_id(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.show import run_show

        console = _make_recording_console()
        rc = asyncio.run(run_show("notHEX!!", console=console))
        assert rc == 1
        out = console.export_text()
        assert "No signal found" in out

    def test_show_cancelled_row_no_payload(
        self, tmp_signals_db, patched_db
    ) -> None:
        from financialagents.cli.show import run_show
        from financialagents.storage.signals import (
            insert_running,
            update_cancelled,
        )

        asyncio.run(run_migrations(tmp_signals_db))
        asyncio.run(
            insert_running(
                "cccc1111", "RELIANCE", "2026-04-26",
                provider="ollama", model="llama3.1:8b",
                db_path=tmp_signals_db,
            )
        )
        asyncio.run(
            update_cancelled(
                "cccc1111",
                error_msg="user cancelled",
                payload_json=None,
                db_path=tmp_signals_db,
            )
        )
        conn = sqlite3.connect(tmp_signals_db)
        try:
            row = conn.execute(
                "SELECT payload_json FROM signals WHERE run_id='cccc1111'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] is None

        console = _make_recording_console()
        rc = asyncio.run(run_show("cccc1111", console=console))
        assert rc == 0
        out = console.export_text()
        assert "cccc1111" in out
        assert "cancelled" in out
