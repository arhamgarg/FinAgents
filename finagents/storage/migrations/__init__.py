from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import aiosqlite

MIGRATIONS_PKG = "finagents.storage.migrations"

_MIGRATIONS_APPLIED: dict[Path, int] = {}


def _utc_iso_safe() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")


def _list_migrations() -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    files = resources.files(MIGRATIONS_PKG)
    pattern = re.compile(r"^(\d+)__(.+)\.sql$")
    for entry in files.iterdir():
        m = pattern.match(entry.name)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), entry.read_text()))
    out.sort(key=lambda t: t[0])
    return out


async def _read_version(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    async with aiosqlite.connect(db_path) as db:
        try:
            async with db.execute("SELECT value FROM _meta WHERE key='schema_version'") as cur:
                row = await cur.fetchone()
        except aiosqlite.OperationalError:
            return 0
    return int(row[0]) if row else 0


async def run_migrations(db_path: Path) -> int:
    if db_path in _MIGRATIONS_APPLIED:
        return _MIGRATIONS_APPLIED[db_path]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    current = await _read_version(db_path)
    all_migrations = _list_migrations()
    pending = [m for m in all_migrations if m[0] > current]

    if pending and db_path.exists():
        backup = db_path.with_name(f"{db_path.stem}-{_utc_iso_safe()}{db_path.suffix}")
        shutil.copy2(db_path, backup)

    async with aiosqlite.connect(db_path) as db:
        for n, _name, sql in pending:
            await db.executescript(sql)
            await db.execute(
                "INSERT OR REPLACE INTO _meta(key, value) VALUES ('schema_version', ?)",
                (str(n),),
            )
            await db.commit()

    new_version = pending[-1][0] if pending else current
    _MIGRATIONS_APPLIED[db_path] = new_version
    return new_version
