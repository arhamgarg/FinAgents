from __future__ import annotations

import json
from datetime import datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel

from finagents.dataflows import cache, rate_limits
from finagents.dataflows.data_result import DataResult

NSE_HOME = "https://www.nseindia.com"
NSE_ALL_INDICES = "https://www.nseindia.com/api/allIndices"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

REQUIRED_SECTORAL = frozenset(
    {
        "NIFTY BANK",
        "NIFTY IT",
        "NIFTY PHARMA",
        "NIFTY METAL",
        "NIFTY AUTO",
        "NIFTY FMCG",
        "NIFTY REALTY",
        "NIFTY ENERGY",
        "NIFTY INFRA",
    }
)

CATEGORY = "sectoral"
SOURCE = "nse"
CACHE_KEY = "allIndices"

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


class SectoralIndexSnapshot(BaseModel):
    index: str
    last: float
    variation: float
    percent_change: float
    open: float
    high: float
    low: float
    previous_close: float
    year_high: Optional[float] = None
    year_low: Optional[float] = None


def is_market_hours(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now(IST)
    if now.tzinfo is None:
        raise ValueError(
            "is_market_hours() received a tz-naive datetime; "
            "pass a tz-aware datetime or None to use current IST time."
        )
    now_ist = now.astimezone(IST)
    if now_ist.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def sectoral_ttl_seconds(now: Optional[datetime] = None) -> int:
    return 300 if is_market_hours(now) else 86400


def _maybe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_indices(payload: dict) -> dict[str, SectoralIndexSnapshot]:
    result: dict[str, SectoralIndexSnapshot] = {}
    for entry in payload.get("data", []):
        symbol = entry.get("indexSymbol", "")
        if symbol not in REQUIRED_SECTORAL:
            continue
        try:
            result[symbol] = SectoralIndexSnapshot(
                index=symbol,
                last=float(entry["last"]),
                variation=float(entry["variation"]),
                percent_change=float(entry["percentChange"]),
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                previous_close=float(entry["previousClose"]),
                year_high=_maybe_float(entry.get("yearHigh")),
                year_low=_maybe_float(entry.get("yearLow")),
            )
        except (KeyError, ValueError, TypeError):
            continue
    return result


async def get_sectoral_indices(
    now: Optional[datetime] = None,
) -> DataResult[dict[str, SectoralIndexSnapshot]]:
    ttl = sectoral_ttl_seconds(now)

    cached = await cache.get_cached(
        CATEGORY, CACHE_KEY, db_path=cache.CACHE_DB_PATH, ttl_override=ttl
    )
    if cached is not None:
        blob, stale = cached
        payload = json.loads(blob)
        parsed = _parse_indices(payload)
        return DataResult[dict[str, SectoralIndexSnapshot]](
            status="ok" if parsed else "empty",
            data=parsed if parsed else None,
            source=SOURCE,
            stale=stale,
        )

    try:
        async with httpx.AsyncClient(
            headers=NSE_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
        ) as client:
            await rate_limits.acquire("www.nseindia.com")
            home_response = await client.get(NSE_HOME)
            home_response.raise_for_status()

            await rate_limits.acquire("www.nseindia.com")
            api_response = await client.get(NSE_ALL_INDICES)
            api_response.raise_for_status()
            payload = api_response.json()
    except Exception as exc:
        return DataResult[dict[str, SectoralIndexSnapshot]](
            status="error", source=SOURCE, error=f"NSE fetch failed: {exc}"
        )

    parsed = _parse_indices(payload)
    if not parsed:
        return DataResult[dict[str, SectoralIndexSnapshot]](status="empty", source=SOURCE)

    await cache.set_cached(
        CATEGORY, CACHE_KEY, json.dumps(payload).encode(), db_path=cache.CACHE_DB_PATH
    )

    return DataResult[dict[str, SectoralIndexSnapshot]](
        status="ok", data=parsed, source=SOURCE, stale=False
    )
