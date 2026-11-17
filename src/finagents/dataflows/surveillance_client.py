from __future__ import annotations

import csv
import io

import httpx

from finagents.dataflows import cache

NSE_SEC_LIST_URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"
CATEGORY = "surveillance"
CACHE_KEY = "nse_sec_list_v1"
T2T_SERIES = frozenset({"BE", "BZ", "ST"})


async def _load_sec_list() -> bytes:
    cached = await cache.get_cached(CATEGORY, CACHE_KEY, db_path=cache.CACHE_DB_PATH)
    if cached is not None:
        data, stale = cached
        if not stale:
            return data

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
    ) as client:
        response = await client.get(
            NSE_SEC_LIST_URL,
            headers={"User-Agent": "FinAgents/1.0"},
        )
        response.raise_for_status()
        body = response.content

    await cache.set_cached(CATEGORY, CACHE_KEY, body, db_path=cache.CACHE_DB_PATH)
    return body


def _parse_sec_list(content: bytes) -> dict[str, dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        symbol = (row.get("Symbol") or "").strip().upper()
        if symbol:
            rows[symbol] = row
    return rows


async def get_surveillance_data(ticker: str) -> tuple[float | None, bool]:
    try:
        content = await _load_sec_list()
        rows = _parse_sec_list(content)
        row = rows.get(ticker.upper())
        if row is None:
            return None, False
        band_str = (row.get("Band") or "").strip()
        circuit_band_pct = None if band_str in {"No Band", ""} else float(band_str) / 100.0
        t2t_flag = (row.get("Series") or "").strip() in T2T_SERIES
        return circuit_band_pct, t2t_flag
    except Exception:
        return None, False
