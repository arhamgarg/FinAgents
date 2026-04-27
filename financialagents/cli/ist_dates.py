from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def ist_today_str() -> str:
    return datetime.now(IST).date().isoformat()
