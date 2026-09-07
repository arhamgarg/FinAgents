from __future__ import annotations

import json
from functools import lru_cache

import httpx

from finagents.dataflows import cache

SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

CATEGORY = "scrip_master"
CACHE_KEY = "nse_master_v1"


class InstrumentNotFound(Exception):
    pass


async def _master_bytes() -> bytes:
    cached = await cache.get_cached(CATEGORY, CACHE_KEY, db_path=cache.CACHE_DB_PATH)
    if cached is not None:
        data, stale = cached
        if not stale:
            return data

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
    ) as client:
        response = await client.get(SCRIP_MASTER_URL)
        response.raise_for_status()
        body = response.content

    await cache.set_cached(CATEGORY, CACHE_KEY, body, db_path=cache.CACHE_DB_PATH)
    return body


async def resolve_token(symbol: str, exchange: str = "NSE") -> str:
    key = (exchange, symbol.upper().replace("-EQ", ""))
    index = _token_index(await _master_bytes())
    if key not in index:
        raise InstrumentNotFound(
            f"Instrument not found for symbol={symbol!r} exchange={exchange!r}"
        )
    return index[key]


async def get_scrip_master() -> list[dict]:
    return json.loads(await _master_bytes())


@lru_cache(maxsize=1)
def _token_index(data: bytes) -> dict[tuple[str, str], str]:
    index = {}
    for entry in json.loads(data):
        token = str(entry.get("token", ""))
        if token:
            key = (entry.get("exch_seg"), str(entry.get("symbol", "")).upper().replace("-EQ", ""))
            index.setdefault(key, token)
    return index
