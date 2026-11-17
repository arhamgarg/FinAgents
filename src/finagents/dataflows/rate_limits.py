from __future__ import annotations

import asyncio
import random
import weakref

from aiolimiter import AsyncLimiter

from finagents.config import Settings

_LIMITERS_BY_LOOP: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, AsyncLimiter]] = (
    weakref.WeakKeyDictionary()
)


def min_delays(settings: Settings) -> dict[str, float]:
    return {
        "screener.in": settings.screener_min_delay,
        "www.moneycontrol.com": settings.moneycontrol_min_delay,
        "economictimes.indiatimes.com": settings.economictimes_min_delay,
        "www.livemint.com": settings.livemint_min_delay,
        "www.nseindia.com": settings.nse_min_delay,
    }


def _get_limiter(host: str, settings: Settings) -> AsyncLimiter:
    period = min_delays(settings)[host]
    loop = asyncio.get_running_loop()
    per_loop = _LIMITERS_BY_LOOP.get(loop)
    if per_loop is None:
        per_loop = {}
        _LIMITERS_BY_LOOP[loop] = per_loop
    limiter = per_loop.get(host)
    if limiter is None:
        limiter = AsyncLimiter(max_rate=1, time_period=period)
        per_loop[host] = limiter
    return limiter


async def acquire(host: str, settings: Settings) -> None:
    limiter = _get_limiter(host, settings)
    async with limiter:
        pass
    await asyncio.sleep(random.uniform(0.0, settings.rate_jitter_max))  # noqa: S311
