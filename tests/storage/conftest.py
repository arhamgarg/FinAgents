from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_signals_db(tmp_path: Path) -> Path:
    return tmp_path / "signals.db"


@pytest.fixture
async def migrated_db(tmp_signals_db: Path) -> Path:
    from finagents.storage.migrations import run_migrations

    await run_migrations(tmp_signals_db)
    return tmp_signals_db
