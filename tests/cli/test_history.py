from __future__ import annotations

import asyncio
import io
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from financialagents.agents.utils.agent_states import AgentState
from financialagents.storage.migrations import run_migrations
from financialagents.storage.signals import insert_running, update_success


def _seed_three_rows(db_path: Path, *, with_payload: bool = False) -> None:
    asyncio.run(run_migrations(db_path))

    seeds = [
        ("aaaa1111", "TCS", "2026-04-25", "2026-04-25T10:00:00+00:00"),
        ("bbbb2222", "INFY", "2026-04-24", "2026-04-24T10:00:00+00:00"),
        ("cccc3333", "RELIANCE", "2026-04-26", "2026-04-26T10:00:00+00:00"),
    ]
    for run_id, ticker, date, created_at in seeds:
        asyncio.run(
            insert_running(
                run_id, ticker, date,
                provider="openrouter", model="openrouter/auto",
                db_path=db_path,
            )
        )
        if with_payload:
            state = _state_with_debate(ticker, date)
            payload = state.model_dump_json(exclude={"messages"})
        else:
            payload = AgentState(ticker=ticker, date=date).model_dump_json(
                exclude={"messages"}
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
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE signals SET created_at=? WHERE run_id=?",
                (created_at, run_id),
            )
            conn.commit()
        finally:
            conn.close()


def _state_with_debate(ticker: str, date: str) -> AgentState:
    from financialagents.agents.managers.schemas import PortfolioDecision
    from financialagents.agents.trader.schemas import TradingSignal

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
        rationale="bull thesis dominates",
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
        pm_notes=f"verdict for {ticker}",
        signal=ts,
    )
    return AgentState(
        ticker=ticker,
        date=date,
        final_trade_decision=pd_,
    )


def _make_recording_console() -> Console:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200, record=True)


@pytest.fixture
def patched_db(monkeypatch, tmp_signals_db):
    monkeypatch.setattr(
        "financialagents.cli.history.signals_db_path",
        lambda settings=None: tmp_signals_db,
    )
    return tmp_signals_db


class TestFaHistory:
    def test_default_table(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.history import run_history

        _seed_three_rows(tmp_signals_db)
        console = _make_recording_console()
        rc = asyncio.run(
            run_history(
                ticker=None, since=None, limit=20,
                show_all=False, expand=False,
                prune=False, before=None,
                yes=False,
                console=console,
            )
        )
        assert rc == 0
        out = console.export_text()
        for header in (
            "created_at", "ticker", "action", "confidence",
            "position_size_pct", "approved", "status",
        ):
            assert header in out, f"missing column header: {header}"
        assert "RELIANCE" in out
        assert "TCS" in out
        assert "INFY" in out
        assert out.index("RELIANCE") < out.index("TCS") < out.index("INFY")

    def test_filters(self, monkeypatch, tmp_signals_db, patched_db) -> None:
        from financialagents.cli import history as history_mod

        captured: dict[str, Any] = {}

        async def _fake_load_history(db_path, *, limit, ticker=None, since=None):
            captured["limit"] = limit
            captured["ticker"] = ticker
            captured["since"] = since
            return []

        monkeypatch.setattr(history_mod, "load_history", _fake_load_history)

        console = _make_recording_console()
        asyncio.run(
            history_mod.run_history(
                ticker="RELIANCE", since="2026-04-01", limit=5,
                show_all=False, expand=False,
                prune=False, before=None,
                yes=False,
                console=console,
            )
        )
        assert captured == {
            "limit": 5,
            "ticker": "RELIANCE",
            "since": "2026-04-01",
        }

        captured.clear()
        asyncio.run(
            history_mod.run_history(
                ticker=None, since=None, limit=20,
                show_all=True, expand=False,
                prune=False, before=None,
                yes=False,
                console=_make_recording_console(),
            )
        )
        assert captured["limit"] == 10_000
        assert captured["ticker"] is None
        assert captured["since"] is None

    def test_prune_non_tty_without_yes_refuses(
        self, monkeypatch, tmp_signals_db, patched_db
    ) -> None:
        from financialagents.cli.history import run_history

        _seed_three_rows(tmp_signals_db)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        console = _make_recording_console()
        rc = asyncio.run(
            run_history(
                ticker=None, since=None, limit=20,
                show_all=False, expand=False,
                prune=True, before="2030-01-01",
                yes=False,
                console=console,
            )
        )
        assert rc == 1
        out = console.export_text()
        assert "Refusing to prune in non-interactive mode without --yes." in out
        conn = sqlite3.connect(tmp_signals_db)
        try:
            n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        finally:
            conn.close()
        assert n == 3

    def test_prune_with_yes_deletes(
        self, monkeypatch, tmp_signals_db, patched_db
    ) -> None:
        from financialagents.cli.history import run_history

        _seed_three_rows(tmp_signals_db)
        console = _make_recording_console()
        rc = asyncio.run(
            run_history(
                ticker=None, since=None, limit=20,
                show_all=False, expand=False,
                prune=True, before="2030-01-01",
                yes=True,
                console=console,
            )
        )
        assert rc == 0
        out = console.export_text()
        assert "Deleted 3 signals." in out
        conn = sqlite3.connect(tmp_signals_db)
        try:
            n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        finally:
            conn.close()
        assert n == 0

    def test_expand(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.history import run_history

        _seed_three_rows(tmp_signals_db, with_payload=True)
        console = _make_recording_console()
        rc = asyncio.run(
            run_history(
                ticker=None, since=None, limit=20,
                show_all=False, expand=True,
                prune=False, before=None,
                yes=False,
                console=console,
            )
        )
        assert rc == 0
        out = console.export_text()
        assert "verdict for RELIANCE" in out
        assert "verdict for TCS" in out
        assert "verdict for INFY" in out

    def test_empty_state(self, tmp_signals_db, patched_db) -> None:
        from financialagents.cli.history import run_history

        console = _make_recording_console()
        rc = asyncio.run(
            run_history(
                ticker=None, since=None, limit=20,
                show_all=False, expand=False,
                prune=False, before=None,
                yes=False,
                console=console,
            )
        )
        assert rc == 0
        out = console.export_text()
        assert (
            "No signals yet. Run `fa run <ticker>` to generate your first signal."
            in out
        )
