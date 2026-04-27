from __future__ import annotations

import asyncio
from pathlib import Path

import platformdirs
import pytest


class TestFADataDirIntegration:
    @pytest.mark.asyncio
    async def test_fa_data_dir_env_end_to_end(
        self,
        clean_env,
        monkeypatch,
        tmp_path: Path,
        fake_pipeline,
    ) -> None:
        target_dir = tmp_path / "fa-data"
        monkeypatch.setenv("FA_DATA_DIR", str(target_dir))

        from financialagents.storage.paths import signals_db_path as real

        monkeypatch.setattr(
            "financialagents.cli.run_command.signals_db_path", real,
        )
        monkeypatch.setattr(
            "financialagents.cli.history.signals_db_path", real,
        )
        monkeypatch.setattr(
            "financialagents.cli.export.signals_db_path", real,
        )

        from financialagents.cli.export import run_export
        from financialagents.cli.history import run_history
        from financialagents.cli.run_command import run_pipeline

        rc = await run_pipeline(
            ticker="RELIANCE",
            date="2026-04-26",
            json_output=True,
            provider_override="ollama",
            debug=False,
        )
        assert rc == 0, f"run_pipeline exited {rc}, expected 0"

        env_db = target_dir / "signals.db"
        assert env_db.exists(), (
            f"fa run did not write signals.db at FA_DATA_DIR={target_dir}; "
            f"contents={list(target_dir.iterdir()) if target_dir.exists() else 'dir missing'}"
        )

        rc = await run_history(
            ticker=None,
            since=None,
            limit=20,
            show_all=False,
            expand=False,
            prune=False,
            before=None,
            yes=False,
        )
        assert rc == 0, f"run_history exited {rc}, expected 0"

        rc = await run_export(
            format_="json",
            ticker=None,
            since=None,
            until=None,
            out=None,
        )
        assert rc == 0, f"run_export exited {rc}, expected 0"

        import sqlite3

        conn = sqlite3.connect(env_db)
        try:
            n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        finally:
            conn.close()
        assert n >= 1, (
            f"FA_DATA_DIR={target_dir} signals.db is empty after run_pipeline"
        )

    def test_fa_data_dir_unset_uses_platformdirs(
        self, clean_env, monkeypatch,
    ) -> None:
        monkeypatch.delenv("FA_DATA_DIR", raising=False)
        from financialagents.config import Settings
        from financialagents.storage.paths import signals_db_path

        p = signals_db_path(Settings(_env_file=None))
        expected = (
            Path(platformdirs.user_data_dir("financialagents", appauthor=False))
            / "signals.db"
        )
        assert p == expected, (
            f"signals_db_path with FA_DATA_DIR unset returned {p}, "
            f"expected platformdirs default {expected}"
        )

    @pytest.mark.asyncio
    async def test_fa_data_dir_empty_dir_graceful(
        self, clean_env, monkeypatch, tmp_path: Path,
    ) -> None:
        target_dir = tmp_path / "fa-empty"
        monkeypatch.setenv("FA_DATA_DIR", str(target_dir))
        from financialagents.storage.paths import signals_db_path as real

        monkeypatch.setattr(
            "financialagents.cli.history.signals_db_path", real,
        )
        monkeypatch.setattr(
            "financialagents.cli.export.signals_db_path", real,
        )

        from financialagents.cli.export import run_export
        from financialagents.cli.history import run_history

        rc = await run_history(
            ticker=None,
            since=None,
            limit=20,
            show_all=False,
            expand=False,
            prune=False,
            before=None,
            yes=False,
        )
        assert rc == 0

        rc = await run_export(
            format_="json",
            ticker=None,
            since=None,
            until=None,
            out=None,
        )
        assert rc == 0
