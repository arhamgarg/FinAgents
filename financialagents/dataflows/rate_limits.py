from __future__ import annotations

import asyncio
import os
import random

from aiolimiter import AsyncLimiter

SCREENER_MIN_DELAY = float(os.getenv("SCREENER_MIN_DELAY", "4.0"))
MONEYCONTROL_MIN_DELAY = float(os.getenv("MONEYCONTROL_MIN_DELAY", "2.0"))
ECONOMICTIMES_MIN_DELAY = float(os.getenv("ECONOMICTIMES_MIN_DELAY", "2.0"))
LIVEMINT_MIN_DELAY = float(os.getenv("LIVEMINT_MIN_DELAY", "2.0"))
NSE_MIN_DELAY = float(os.getenv("NSE_MIN_DELAY", "1.0"))
STOCKTWITS_MIN_DELAY = float(os.getenv("STOCKTWITS_MIN_DELAY", "20.0"))
GOOGLE_TRENDS_MIN_DELAY = float(os.getenv("GOOGLE_TRENDS_MIN_DELAY", "5.0"))
JITTER_MAX = float(os.getenv("RATE_JITTER_MAX", "2.0"))

_LIMITERS: dict[str, AsyncLimiter] = {
    "screener.in": AsyncLimiter(max_rate=1, time_period=SCREENER_MIN_DELAY),
    "www.moneycontrol.com": AsyncLimiter(max_rate=1, time_period=MONEYCONTROL_MIN_DELAY),
    "economictimes.indiatimes.com": AsyncLimiter(max_rate=1, time_period=ECONOMICTIMES_MIN_DELAY),
    "www.livemint.com": AsyncLimiter(max_rate=1, time_period=LIVEMINT_MIN_DELAY),
    "www.nseindia.com": AsyncLimiter(max_rate=1, time_period=NSE_MIN_DELAY),
    "api.stocktwits.com": AsyncLimiter(max_rate=1, time_period=STOCKTWITS_MIN_DELAY),
    "trends.google.com": AsyncLimiter(max_rate=1, time_period=GOOGLE_TRENDS_MIN_DELAY),
}


async def acquire(host: str) -> None:
    limiter = _LIMITERS[host]
    async with limiter:
        pass
    await asyncio.sleep(random.uniform(0.0, JITTER_MAX))
