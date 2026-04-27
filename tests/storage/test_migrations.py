from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from financialagents.storage import migrations as migrations_mod
from financialagents.storage.migrations import (
    _MIGRATIONS_APPLIED,
    run_migrations,
)


@pytest.fixture(autouse=True)
def reset_cache():
    _MIGRATIONS_APPLIED.clear()
    yield
    _MIGRATIONS_APPLIED.clear()


class TestMigrations:
    @pytest.mark.asyncio
    async def test_fresh_install(self, tmp_path: Path) -> None:
        db = tmp_path / "signals.db"
        v = await run_migrations(db)
        assert v == 1
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT value FROM _meta WHERE key='schema_version'"
            ).fetchone()
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "1"
        assert "signals" in tables
        assert "_meta" in tables

    @pytest.mark.asyncio
    async def test_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = tmp_path / "signals.db"

        counter = {"n": 0}
        real_read_version = migrations_mod._read_version

        async def counted_read_version(path: Path) -> int:
            counter["n"] += 1
            return await real_read_version(path)

        monkeypatch.setattr(
            migrations_mod, "_read_version", counted_read_version,
        )

        v1 = await run_migrations(db)
        v2 = await run_migrations(db)

        assert v1 == v2 == 1
        assert counter["n"] == 1, (
            f"expected _read_version called exactly once across two "
            f"run_migrations() calls (cache hit on second), got {counter['n']}"
        )

    @pytest.mark.asyncio
    async def test_no_backup_on_fresh_install(self, tmp_path: Path) -> None:
        db = tmp_path / "signals.db"
        await run_migrations(db)
        backups = [
            p
            for p in tmp_path.iterdir()
            if p.name != "signals.db" and p.suffix == ".db"
        ]
        assert backups == []
