from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import aiosqlite

from finagents.clock import utc_now

CACHE_DB_PATH = Path(".cache/finagents.db")

Category = Literal["scrip_master", "ohlcv", "fundamentals", "news", "surveillance"]

TTL_SECONDS: dict[str, int] = {
    "scrip_master": 86400,
    "ohlcv": 86400,
    "fundamentals": 86400,
    "news": 1800,
    "surveillance": 86400,
}

_VALID_CATEGORIES = frozenset(TTL_SECONDS.keys())


def _check_category(category: str) -> None:
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"Unknown cache category: {category!r}")


async def init_cache(db_path: Path = CACHE_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await _create_tables(db)
        await db.commit()


async def _create_tables(db: aiosqlite.Connection) -> None:
    for category in _VALID_CATEGORIES:
        await db.execute(
            f"CREATE TABLE IF NOT EXISTS {category} ("
            "key TEXT PRIMARY KEY, value BLOB NOT NULL, fetched_at TEXT NOT NULL)"
        )


async def get_cached(
    category: str,
    key: str,
    db_path: Path = CACHE_DB_PATH,
    ttl_override: int | None = None,
) -> tuple[bytes, bool] | None:
    _check_category(category)
    if not db_path.exists():
        return None
    ttl = ttl_override if ttl_override is not None else TTL_SECONDS[category]
    async with (
        aiosqlite.connect(db_path) as db,
        db.execute(
            f"SELECT value, fetched_at FROM {category} WHERE key = ?",  # noqa: S608
            (key,),
        ) as cursor,
    ):
        row = await cursor.fetchone()
    if row is None:
        return None
    fetched_at = datetime.fromisoformat(row[1])
    age = (utc_now() - fetched_at).total_seconds()
    return (row[0], age > ttl)


async def set_cached(
    category: str,
    key: str,
    value: bytes,
    db_path: Path = CACHE_DB_PATH,
) -> None:
    _check_category(category)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await _create_tables(db)
        await db.execute(
            f"INSERT OR REPLACE INTO {category} (key, value, fetched_at) VALUES (?, ?, ?)",  # noqa: S608
            (key, value, utc_now().isoformat()),
        )
        await db.commit()
