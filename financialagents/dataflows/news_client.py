from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from financialagents.dataflows import cache, rate_limits
from financialagents.dataflows._ticker import sanitize_ticker
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.news_schema import NewsArticle

MONEYCONTROL_URL = "https://www.moneycontrol.com/news/tags/{ticker}/"
ECONOMIC_TIMES_URL = "https://economictimes.indiatimes.com/topic/{ticker}"
LIVEMINT_URL = "https://www.livemint.com/Search/Link/Keyword/{ticker}"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

CATEGORY = "news"
MAX_PER_SOURCE = 10


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower()


def _dedupe(articles: list[NewsArticle]) -> list[NewsArticle]:
    seen: set[str] = set()
    out: list[NewsArticle] = []
    for a in articles:
        k = _normalize_url(a.url)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def _parse_moneycontrol(html_bytes: bytes) -> list[NewsArticle]:
    soup = BeautifulSoup(html_bytes, "lxml")
    items: list[NewsArticle] = []
    for li in soup.find_all("li", class_="clearfix")[:MAX_PER_SOURCE]:
        link = li.find("a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        snippet_el = li.find("p")
        snippet = snippet_el.get_text(strip=True) if snippet_el else None
        items.append(NewsArticle(
            title=title,
            url=link["href"],
            source="moneycontrol",
            snippet=snippet,
        ))
    return items


def _parse_economic_times(html_bytes: bytes) -> list[NewsArticle]:
    soup = BeautifulSoup(html_bytes, "lxml")
    items: list[NewsArticle] = []
    for div in soup.find_all("div", class_="eachStory")[:MAX_PER_SOURCE]:
        link = div.find("a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        snippet_el = div.find("p")
        snippet = snippet_el.get_text(strip=True) if snippet_el else None
        published: Optional[datetime] = None
        time_el = div.find("time")
        if time_el and time_el.get("datetime"):
            try:
                published = datetime.fromisoformat(time_el["datetime"])
            except ValueError:
                published = None
        items.append(NewsArticle(
            title=title,
            url=link["href"],
            source="economictimes",
            snippet=snippet,
            published_at=published,
        ))
    return items


def _parse_livemint(html_bytes: bytes) -> list[NewsArticle]:
    soup = BeautifulSoup(html_bytes, "lxml")
    items: list[NewsArticle] = []
    for article in soup.find_all("article")[:MAX_PER_SOURCE]:
        link = article.find("a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        snippet_el = article.find("p", class_="summary") or article.find("p")
        snippet = snippet_el.get_text(strip=True) if snippet_el else None
        items.append(NewsArticle(
            title=title,
            url=link["href"],
            source="livemint",
            snippet=snippet,
        ))
    return items


async def _fetch_source(
    client: httpx.AsyncClient,
    url: str,
    host: str,
    parser,
) -> list[NewsArticle]:
    await rate_limits.acquire(host)
    response = await client.get(url, headers=_DEFAULT_HEADERS)
    response.raise_for_status()
    return parser(response.content)


async def get_news(ticker: str) -> DataResult[list[NewsArticle]]:
    try:
        safe_ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        return DataResult[list[NewsArticle]](
            status="error", source="news", error=str(exc)
        )

    cached = await cache.get_cached(CATEGORY, safe_ticker, db_path=cache.CACHE_DB_PATH)
    if cached is not None:
        blob, stale = cached
        payload = json.loads(blob)
        if isinstance(payload, list):
            data = [NewsArticle.model_validate(a) for a in payload]
            cached_status = "ok"
        else:
            data = [NewsArticle.model_validate(a) for a in payload.get("articles", [])]
            cached_status = payload.get("status", "ok")
        return DataResult[list[NewsArticle]](
            status=cached_status, data=data, source="news_cache", stale=stale
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(
            _fetch_source(
                client,
                MONEYCONTROL_URL.format(ticker=safe_ticker.lower()),
                "www.moneycontrol.com",
                _parse_moneycontrol,
            ),
            _fetch_source(
                client,
                ECONOMIC_TIMES_URL.format(ticker=safe_ticker.lower()),
                "economictimes.indiatimes.com",
                _parse_economic_times,
            ),
            _fetch_source(
                client,
                LIVEMINT_URL.format(ticker=safe_ticker.lower()),
                "www.livemint.com",
                _parse_livemint,
            ),
            return_exceptions=True,
        )

    collected: list[NewsArticle] = []
    successes = 0
    errors: list[str] = []
    source_names = ["moneycontrol", "economictimes", "livemint"]
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            errors.append(f"{name}: {result}")
            continue
        successes += 1
        collected.extend(result)

    if successes == 0:
        return DataResult[list[NewsArticle]](
            status="error",
            source="news",
            error="; ".join(errors) or "no sources succeeded",
        )

    deduped = _dedupe(collected)

    if successes >= 2:
        cache_payload = {
            "status": "ok",
            "articles": [a.model_dump(mode="json") for a in deduped],
        }
        await cache.set_cached(
            CATEGORY,
            safe_ticker,
            json.dumps(cache_payload).encode(),
            db_path=cache.CACHE_DB_PATH,
        )
        return DataResult[list[NewsArticle]](
            status="ok", data=deduped, source="news"
        )

    return DataResult[list[NewsArticle]](
        status="partial", data=deduped, source="news",
        error=f"Only 1/3 sources succeeded: {'; '.join(errors)}",
    )
