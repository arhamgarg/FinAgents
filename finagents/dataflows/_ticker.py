from __future__ import annotations

import re

_TICKER_PATTERN = re.compile(r"^[A-Z0-9&\-]{1,20}$")


def sanitize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    t = ticker.upper()
    if not _TICKER_PATTERN.fullmatch(t):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return t
