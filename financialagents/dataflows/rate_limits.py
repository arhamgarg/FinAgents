from __future__ import annotations

import asyncio
import os
import random
import weakref

from aiolimiter import AsyncLimiter

SCREENER_MIN_DELAY = float(os.getenv("SCREENER_MIN_DELAY", "4.0"))
MONEYCONTROL_MIN_DELAY = float(os.getenv("MONEYCONTROL_MIN_DELAY", "2.0"))
ECONOMICTIMES_MIN_DELAY = float(os.getenv("ECONOMICTIMES_MIN_DELAY", "2.0"))
LIVEMINT_MIN_DELAY = float(os.getenv("LIVEMINT_MIN_DELAY", "2.0"))
NSE_MIN_DELAY = float(os.getenv("NSE_MIN_DELAY", "1.0"))
GOOGLE_TRENDS_MIN_DELAY = float(os.getenv("GOOGLE_TRENDS_MIN_DELAY", "5.0"))
JITTER_MAX = float(os.getenv("RATE_JITTER_MAX", "2.0"))

_HOST_CONFIG: dict[str, tuple[int, float]] = {
    "screener.in": (1, SCREENER_MIN_DELAY),
    "www.moneycontrol.com": (1, MONEYCONTROL_MIN_DELAY),
    "economictimes.indiatimes.com": (1, ECONOMICTIMES_MIN_DELAY),
    "www.livemint.com": (1, LIVEMINT_MIN_DELAY),
    "www.nseindia.com": (1, NSE_MIN_DELAY),
    "trends.google.com": (1, GOOGLE_TRENDS_MIN_DELAY),
 }

_LIMITERS_BY_LOOP: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, AsyncLimiter]
] = weakref.WeakKeyDictionary()


def _get_limiter(host: str) -> AsyncLimiter:
    rate, period = _HOST_CONFIG[host]
    loop = asyncio.get_running_loop()
    per_loop = _LIMITERS_BY_LOOP.get(loop)
    if per_loop is None:
        per_loop = {}
        _LIMITERS_BY_LOOP[loop] = per_loop
    limiter = per_loop.get(host)
    if limiter is None:
        limiter = AsyncLimiter(max_rate=rate, time_period=period)
        per_loop[host] = limiter
    return limiter


async def acquire(host: str) -> None:
    limiter = _get_limiter(host)
    async with limiter:
        pass
    await asyncio.sleep(random.uniform(0.0, JITTER_MAX))
