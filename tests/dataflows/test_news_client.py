from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pytest_httpx import HTTPXMock

from financialagents.dataflows.news_client import (
    _parse_economic_times,
    _parse_livemint,
    _parse_moneycontrol,
    get_news,
)
from financialagents.dataflows.news_schema import NewsArticle


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "news"

MC_URL = "https://www.moneycontrol.com/news/tags/reliance/"
ET_URL = "https://economictimes.indiatimes.com/topic/reliance"
LM_URL = "https://www.livemint.com/Search/Link/Keyword/reliance"




def test_moneycontrol_parser() -> None:
    html = (FIXTURE_DIR / "moneycontrol_reliance.html").read_bytes()
    articles = _parse_moneycontrol(html)
    assert len(articles) == 3
    assert articles[0].source == "moneycontrol"
    assert "Q3 profit" in articles[0].title


def test_economictimes_parser() -> None:
    html = (FIXTURE_DIR / "economictimes_reliance.html").read_bytes()
    articles = _parse_economic_times(html)
    assert len(articles) == 2
    assert articles[0].source == "economictimes"
    assert articles[0].published_at is not None


def test_livemint_parser() -> None:
    html = (FIXTURE_DIR / "livemint_reliance.html").read_bytes()
    articles = _parse_livemint(html)
    assert len(articles) == 2
    assert articles[0].source == "livemint"




@pytest.mark.asyncio
async def test_aggregator_ok_when_3_sources_succeed(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        httpx_mock.add_response(url=MC_URL, content=(FIXTURE_DIR / "moneycontrol_reliance.html").read_bytes())
        httpx_mock.add_response(url=ET_URL, content=(FIXTURE_DIR / "economictimes_reliance.html").read_bytes())
        httpx_mock.add_response(url=LM_URL, content=(FIXTURE_DIR / "livemint_reliance.html").read_bytes())
        result = await get_news("RELIANCE")

    assert result.status == "ok"
    assert result.data is not None
    assert len(result.data) >= 5
    sources_in_result = {a.source for a in result.data}
    assert sources_in_result == {"moneycontrol", "economictimes", "livemint"}


@pytest.mark.asyncio
async def test_aggregator_ok_when_2_sources_succeed(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        httpx_mock.add_response(url=MC_URL, status_code=503)
        httpx_mock.add_response(url=ET_URL, content=(FIXTURE_DIR / "economictimes_reliance.html").read_bytes())
        httpx_mock.add_response(url=LM_URL, content=(FIXTURE_DIR / "livemint_reliance.html").read_bytes())
        result = await get_news("RELIANCE")

    assert result.status == "ok"
    assert result.data is not None


@pytest.mark.asyncio
async def test_aggregator_partial_when_1_source_succeeds(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        httpx_mock.add_response(url=MC_URL, status_code=503)
        httpx_mock.add_response(url=ET_URL, status_code=503)
        httpx_mock.add_response(url=LM_URL, content=(FIXTURE_DIR / "livemint_reliance.html").read_bytes())
        result = await get_news("RELIANCE")

    assert result.status == "partial"
    assert result.data is not None
    assert all(a.source == "livemint" for a in result.data)


@pytest.mark.asyncio
async def test_aggregator_error_when_0_sources_succeed(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        httpx_mock.add_response(url=MC_URL, status_code=503)
        httpx_mock.add_response(url=ET_URL, status_code=503)
        httpx_mock.add_response(url=LM_URL, status_code=503)
        result = await get_news("RELIANCE")

    assert result.status == "error"
    assert result.error is not None
    assert "moneycontrol" in result.error
    assert "economictimes" in result.error
    assert "livemint" in result.error


@pytest.mark.asyncio
async def test_dedup_across_sources(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financialagents.dataflows.news_client import _dedupe

    articles = [
        NewsArticle(
            title="Same story on MC",
            url="https://www.moneycontrol.com/news/reliance-story.html",
            source="moneycontrol",
        ),
        NewsArticle(
            title="Same story on ET",
            url="https://www.moneycontrol.com/news/reliance-story.html",
            source="economictimes",
        ),
        NewsArticle(
            title="Unique story",
            url="https://www.livemint.com/companies/news/unique-story.html",
            source="livemint",
        ),
    ]
    deduped = _dedupe(articles)
    assert len(deduped) == 2


@pytest.mark.asyncio
async def test_ticker_sanitization_rejects_injection(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    result = await get_news("RELIANCE/../etc")
    assert result.status == "error"
    assert result.error is not None
    assert "Invalid ticker" in result.error


@pytest.mark.asyncio
async def test_rate_limiter_per_domain(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", tmp_path / "cache.db")
    acquire_mock = AsyncMock()
    with patch("financialagents.dataflows.news_client.rate_limits") as mock_rl:
        mock_rl.acquire = acquire_mock
        httpx_mock.add_response(url=MC_URL, content=(FIXTURE_DIR / "moneycontrol_reliance.html").read_bytes())
        httpx_mock.add_response(url=ET_URL, content=(FIXTURE_DIR / "economictimes_reliance.html").read_bytes())
        httpx_mock.add_response(url=LM_URL, content=(FIXTURE_DIR / "livemint_reliance.html").read_bytes())
        await get_news("RELIANCE")

    call_args = [call.args[0] for call in acquire_mock.call_args_list]
    assert "www.moneycontrol.com" in call_args
    assert "economictimes.indiatimes.com" in call_args
    assert "www.livemint.com" in call_args
    assert acquire_mock.call_count == 3


@pytest.mark.asyncio
async def test_cache_hit(
    httpx_mock: HTTPXMock, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "cache.db"
    monkeypatch.setattr("financialagents.dataflows.cache.CACHE_DB_PATH", cache_path)
    monkeypatch.setattr("financialagents.dataflows.news_client.cache.CACHE_DB_PATH", cache_path)

    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        httpx_mock.add_response(url=MC_URL, content=(FIXTURE_DIR / "moneycontrol_reliance.html").read_bytes())
        httpx_mock.add_response(url=ET_URL, content=(FIXTURE_DIR / "economictimes_reliance.html").read_bytes())
        httpx_mock.add_response(url=LM_URL, content=(FIXTURE_DIR / "livemint_reliance.html").read_bytes())
        first = await get_news("RELIANCE")

    assert first.status == "ok"

    with patch("financialagents.dataflows.rate_limits.acquire", new_callable=AsyncMock):
        second = await get_news("RELIANCE")

    assert second.status == "ok"
    assert second.data is not None
