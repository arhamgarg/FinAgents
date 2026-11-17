from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def local_now() -> datetime:
    return datetime.now().astimezone()


def ist_today_str() -> str:
    return datetime.now(IST).date().isoformat()
