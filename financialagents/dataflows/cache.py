from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import aiosqlite

CACHE_DB_PATH = Path(".cache/financialagents.db")

Category = Literal["scrip_master", "ohlcv", "fundamentals", "news", "sentiment", "sectoral"]

TTL_SECONDS: dict[str, int] = {
    "scrip_master": 86400,
    "ohlcv": 86400,
    "fundamentals": 86400,
    "news": 1800,
    "sentiment": 900,
    "sectoral": 300,
}

_VALID_CATEGORIES = frozenset(TTL_SECONDS.keys())


def _check_category(category: str) -> None:
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"Unknown cache category: {category!r}")


async def init_cache(db_path: Path = CACHE_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        for category in _VALID_CATEGORIES:
            await db.execute(
                f"CREATE TABLE IF NOT EXISTS {category} ("
                "key TEXT PRIMARY KEY, value BLOB NOT NULL, fetched_at TEXT NOT NULL)"
            )
        await db.commit()


async def get_cached(
    category: str,
    key: str,
    db_path: Path = CACHE_DB_PATH,
    ttl_override: Optional[int] = None,
) -> Optional[tuple[bytes, bool]]:
    _check_category(category)
    if not db_path.exists():
        return None
    ttl = ttl_override if ttl_override is not None else TTL_SECONDS[category]
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            f"SELECT value, fetched_at FROM {category} WHERE key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    fetched_at = datetime.fromisoformat(row[1])
    age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
    return (row[0], age > ttl)


async def set_cached(
    category: str,
    key: str,
    value: bytes,
    db_path: Path = CACHE_DB_PATH,
) -> None:
    _check_category(category)
    await init_cache(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"INSERT OR REPLACE INTO {category} (key, value, fetched_at) VALUES (?, ?, ?)",
            (key, value, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
