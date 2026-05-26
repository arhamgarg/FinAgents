from __future__ import annotations

import asyncio
import csv as _csv
import io
import json
import sqlite3
from pathlib import Path

import pytest

from financialagents.agents.utils.agent_states import AgentState
from financialagents.storage.migrations import run_migrations
from financialagents.storage.signals import insert_running, update_success


def _seed_three_rows(db_path: Path) -> None:
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


@pytest.fixture
def patched_db(monkeypatch, tmp_signals_db):
    monkeypatch.setattr(
        "financialagents.cli.export.signals_db_path",
        lambda settings=None: tmp_signals_db,
    )
    return tmp_signals_db


class TestFaExport:
    def test_json(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)

        rc = asyncio.run(
            run_export(
                format_="json",
                ticker=None,
                since="2026-04-01",
                until=None,
                out=None,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        tickers = [r["ticker"] for r in parsed]
        assert tickers == ["RELIANCE", "TCS", "INFY"]
        for row in parsed:
            for col in (
                "run_id", "ticker", "date", "created_at",
                "provider", "model", "status", "action",
                "confidence", "position_size_pct", "approved",
                "pm_decision", "error_msg",
            ):
                assert col in row, f"missing column in JSON row: {col}"

    def test_csv(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)

        rc = asyncio.run(
            run_export(
                format_="csv",
                ticker=None,
                since=None,
                until=None,
                out=None,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        reader = _csv.DictReader(io.StringIO(captured.out))
        rows = list(reader)
        assert len(rows) == 3
        assert reader.fieldnames is not None
        for col in (
            "run_id", "ticker", "date", "created_at",
            "provider", "model", "status", "action",
            "confidence", "position_size_pct", "approved",
            "pm_decision", "error_msg",
        ):
            assert col in reader.fieldnames, f"missing CSV header: {col}"
        assert [r["ticker"] for r in rows] == ["RELIANCE", "TCS", "INFY"]

    def test_no_rows_match(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)

        rc = asyncio.run(
            run_export(
                format_="json",
                ticker=None,
                since="2030-01-01",
                until=None,
                out=None,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert (
            "No signals match the given filters. Showing 0 rows."
            in captured.err
        )
        assert json.loads(captured.out) == []

    def test_no_rows_match_csv_header_only(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)

        rc = asyncio.run(
            run_export(
                format_="csv",
                ticker="DOES_NOT_EXIST",
                since=None,
                until=None,
                out=None,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert (
            "No signals match the given filters. Showing 0 rows."
            in captured.err
        )
        reader = _csv.DictReader(io.StringIO(captured.out))
        assert list(reader) == []
        assert reader.fieldnames is not None
        assert "ticker" in reader.fieldnames

    def test_until_filter(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)

        rc = asyncio.run(
            run_export(
                format_="json",
                ticker=None,
                since=None,
                until="2026-04-26",
                out=None,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        tickers = {r["ticker"] for r in parsed}
        assert tickers == {"TCS", "INFY"}, (
            f"--until 2026-04-26 should exclude RELIANCE, got {tickers}"
        )

    def test_invalid_format(
        self, monkeypatch, capsys, tmp_signals_db, patched_db,
    ) -> None:
        from financialagents.cli.export import run_export

        rc = asyncio.run(
            run_export(
                format_="xml",
                ticker=None,
                since=None,
                until=None,
                out=None,
            )
        )
        assert rc == 1
        captured = capsys.readouterr()
        assert "Unsupported format: xml" in captured.err
        assert "json or csv" in captured.err

    def test_out_file_overwrite_refused_non_tty(
        self, monkeypatch, tmp_signals_db, patched_db, tmp_path,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)
        target = tmp_path / "existing.json"
        target.write_text("PRIOR CONTENT")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = asyncio.run(
            run_export(
                format_="json",
                ticker=None,
                since=None,
                until=None,
                out=str(target),
                yes=False,
            )
        )
        assert rc == 1
        assert target.read_text() == "PRIOR CONTENT"

    def test_out_file_overwrite_with_yes(
        self, monkeypatch, tmp_signals_db, patched_db, tmp_path,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)
        target = tmp_path / "existing.json"
        target.write_text("PRIOR CONTENT")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = asyncio.run(
            run_export(
                format_="json",
                ticker=None,
                since=None,
                until=None,
                out=str(target),
                yes=True,
            )
        )
        assert rc == 0
        parsed = json.loads(target.read_text())
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_out_file_new_path(
        self, monkeypatch, tmp_signals_db, patched_db, tmp_path,
    ) -> None:
        from financialagents.cli.export import run_export

        _seed_three_rows(tmp_signals_db)
        target = tmp_path / "new" / "fresh.csv"
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = asyncio.run(
            run_export(
                format_="csv",
                ticker=None,
                since=None,
                until=None,
                out=str(target),
                yes=False,
            )
        )
        assert rc == 0
        assert target.exists()
        reader = _csv.DictReader(io.StringIO(target.read_text()))
        rows = list(reader)
        assert len(rows) == 3


class TestLoadHistoryUntil:

    def test_until_excludes_rows_at_or_after(self, tmp_signals_db) -> None:
        from financialagents.storage.signals import load_history

        _seed_three_rows(tmp_signals_db)

        rows = asyncio.run(
            load_history(tmp_signals_db, limit=20, until="2026-04-26")
        )
        tickers = {r["ticker"] for r in rows}
        assert tickers == {"TCS", "INFY"}, (
            f"--until 2026-04-26 should exclude RELIANCE, got {tickers}"
        )

    def test_until_composes_with_since_and_ticker(
        self, tmp_signals_db,
    ) -> None:
        from financialagents.storage.signals import load_history

        _seed_three_rows(tmp_signals_db)

        rows = asyncio.run(
            load_history(
                tmp_signals_db,
                limit=20,
                ticker="TCS",
                since="2026-04-01",
                until="2026-04-26",
            )
        )
        assert len(rows) == 1
        assert rows[0]["ticker"] == "TCS"

    def test_until_none_preserves_legacy_behavior(
        self, tmp_signals_db,
    ) -> None:
        from financialagents.storage.signals import load_history

        _seed_three_rows(tmp_signals_db)

        rows = asyncio.run(
            load_history(tmp_signals_db, limit=20, until=None)
        )
        assert len(rows) == 3
