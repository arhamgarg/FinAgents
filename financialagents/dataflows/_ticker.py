from __future__ import annotations

import re

_TICKER_PATTERN = re.compile(r"^[A-Z0-9\-&]{1,20}$")


def sanitize_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_PATTERN.match(t):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return t
