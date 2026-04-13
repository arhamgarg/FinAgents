from __future__ import annotations

import json

import httpx

from financialagents.dataflows import cache

SCRIP_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)

CATEGORY = "scrip_master"
CACHE_KEY = "nse_master_v1"


class InstrumentNotFound(Exception):
    pass


async def get_scrip_master() -> list[dict]:
    cached = await cache.get_cached(CATEGORY, CACHE_KEY, db_path=cache.CACHE_DB_PATH)
    if cached is not None:
        data, stale = cached
        if not stale:
            return json.loads(data)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
    ) as client:
        response = await client.get(SCRIP_MASTER_URL)
        response.raise_for_status()
        body = response.content

    await cache.set_cached(CATEGORY, CACHE_KEY, body, db_path=cache.CACHE_DB_PATH)
    return json.loads(body)


async def resolve_token(symbol: str, exchange: str = "NSE") -> str:
    master = await get_scrip_master()
    normalized = symbol.upper().replace("-EQ", "")
    for entry in master:
        if entry.get("exch_seg") != exchange:
            continue
        entry_symbol = str(entry.get("symbol", "")).upper().replace("-EQ", "")
        if entry_symbol == normalized:
            token = str(entry.get("token", ""))
            if not token:
                continue
            return token
    raise InstrumentNotFound(
        f"Instrument not found for symbol={symbol!r} exchange={exchange!r}"
    )
