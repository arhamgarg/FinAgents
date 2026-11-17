from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

from finagents._redaction import redact_secrets
from finagents.config import Settings
from finagents.dataflows import cache, scrip_master
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.scrip_master import InstrumentNotFoundError
from finagents.dataflows.session_manager import (
    AngelOneAuthError,
    get_session_manager,
)

Interval = Literal[
    "ONE_MINUTE",
    "THREE_MINUTE",
    "FIVE_MINUTE",
    "TEN_MINUTE",
    "FIFTEEN_MINUTE",
    "THIRTY_MINUTE",
    "ONE_HOUR",
    "ONE_DAY",
]

MAX_CANDLES_PER_REQUEST = 500

CANDLES_PER_DAY: dict[str, int] = {
    "ONE_MINUTE": 375,
    "THREE_MINUTE": 125,
    "FIVE_MINUTE": 75,
    "TEN_MINUTE": 38,
    "FIFTEEN_MINUTE": 25,
    "THIRTY_MINUTE": 13,
    "ONE_HOUR": 7,
    "ONE_DAY": 1,
}

CATEGORY = "ohlcv"
SOURCE = "angelone"
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _split_date_range(
    start: datetime, end: datetime, interval: Interval
) -> list[tuple[datetime, datetime]]:
    if interval == "ONE_DAY":
        return [(start, end)]

    candles_per_day = CANDLES_PER_DAY[interval]
    days_per_chunk = max(1, MAX_CANDLES_PER_REQUEST // candles_per_day)

    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=days_per_chunk), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(minutes=1)
    return chunks


def _cache_key(ticker: str, start: datetime, end: datetime, interval: str) -> str:
    raw = f"{ticker.upper()}:{start.date().isoformat()}:{end.date().isoformat()}:{interval}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:32]


def _candles_to_dataframe(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


async def get_ohlcv(  # noqa: PLR0911
    ticker: str,
    start: datetime,
    end: datetime,
    interval: Interval = "ONE_DAY",
    *,
    settings: Settings,
) -> DataResult[pd.DataFrame]:
    key = _cache_key(ticker, start, end, interval)

    cached = await cache.get_cached(CATEGORY, key, db_path=cache.CACHE_DB_PATH)
    if cached is not None:
        blob, stale = cached
        df = pd.read_parquet(io.BytesIO(blob))
        return DataResult[pd.DataFrame](status="ok", data=df, source=SOURCE, stale=stale)

    try:
        token = await scrip_master.resolve_token(ticker, exchange="NSE")
    except InstrumentNotFoundError as exc:
        return DataResult[pd.DataFrame](
            status="error", source=SOURCE, error=f"Instrument not found: {redact_secrets(str(exc))}"
        )
    except Exception as exc:
        return DataResult[pd.DataFrame](
            status="error", source=SOURCE, error=f"Scrip master error: {redact_secrets(str(exc))}"
        )

    sm = get_session_manager(settings)
    all_rows: list[list] = []
    try:
        for chunk_start, chunk_end in _split_date_range(start, end, interval):
            param = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": chunk_start.strftime("%Y-%m-%d %H:%M"),
                "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
            }
            response = await sm.call_with_retry("getCandleData", param)
            rows = response.get("data") or []
            all_rows.extend(rows)
    except AngelOneAuthError as exc:
        return DataResult[pd.DataFrame](
            status="error", source=SOURCE, error=f"Angel One auth: {redact_secrets(str(exc))}"
        )
    except Exception as exc:
        return DataResult[pd.DataFrame](
            status="error",
            source=SOURCE,
            error=f"Angel One call failed: {redact_secrets(str(exc))}",
        )

    if not all_rows:
        return DataResult[pd.DataFrame](status="empty", source=SOURCE)

    df = _candles_to_dataframe(all_rows)

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    await cache.set_cached(CATEGORY, key, buf.getvalue(), db_path=cache.CACHE_DB_PATH)

    return DataResult[pd.DataFrame](status="ok", data=df, source=SOURCE)
