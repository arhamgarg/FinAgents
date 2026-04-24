from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from financialagents.dataflows.cache import (
    TTL_SECONDS,
    get_cached,
    init_cache,
    set_cached,
)


class TestCache:
    @pytest.mark.asyncio
    async def test_set_and_get_returns_value_not_stale(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        key = "RELIANCE:2024-01-01:2024-01-31"
        value = b"some-bytes"
        await set_cached("ohlcv", key, value, db_path=db)
        result = await get_cached("ohlcv", key, db_path=db)
        assert result is not None
        got_value, stale = result
        assert got_value == value
        assert stale is False

    @pytest.mark.asyncio
    async def test_stale_entry_past_ttl(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        key = "RELIANCE:2024-01-01:2024-01-31"
        value = b"stale-bytes"

        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        await init_cache(db)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT OR REPLACE INTO ohlcv (key, value, fetched_at) VALUES (?, ?, ?)",
            (key, value, old_time),
        )
        conn.commit()
        conn.close()

        result = await get_cached("ohlcv", key, db_path=db)
        assert result is not None
        got_value, stale = result
        assert got_value == value
        assert stale is True

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        result = await get_cached("news", "nonexistent-key", db_path=db)
        assert result is None

    @pytest.mark.asyncio
    async def test_init_cache_creates_all_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        await init_cache(db)
        conn = sqlite3.connect(db)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        expected = {"scrip_master", "ohlcv", "fundamentals", "news", "surveillance"}
        assert expected.issubset(tables)

    @pytest.mark.asyncio
    async def test_unknown_category_raises_value_error(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with pytest.raises(ValueError, match="Unknown cache category"):
            await get_cached("bogus", "k", db_path=db)

    def test_sql_uses_parameterized_placeholders(self) -> None:
        import inspect

        import financialagents.dataflows.cache as cache_module

        source = inspect.getsource(cache_module)
        assert "WHERE key = ?" in source or "(key, value, fetched_at) VALUES (?, ?, ?)" in source
        assert 'f"SELECT' not in source.replace('f"SELECT value, fetched_at FROM', "ALLOWED")
        assert "(key," in source

    @pytest.mark.asyncio
    async def test_news_ttl_30_minutes_stale_after_31_min(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        key = "news-key-31min"
        value = b"news-bytes"

        old_time = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
        await init_cache(db)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT OR REPLACE INTO news (key, value, fetched_at) VALUES (?, ?, ?)",
            (key, value, old_time),
        )
        conn.commit()
        conn.close()

        result = await get_cached("news", key, db_path=db)
        assert result is not None
        got_value, stale = result
        assert got_value == value
        assert stale is True
        assert TTL_SECONDS["news"] == 1800
