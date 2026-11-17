from __future__ import annotations

import json

import httpx

from finagents._redaction import redact_secrets
from finagents.config import Settings
from finagents.dataflows import cache, rate_limits
from finagents.dataflows._ticker import sanitize_ticker as _sanitize_ticker
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.fundamentals_schema import (
    DebtBalance,
    FundamentalsReport,
    Profitability,
    Shareholding,
    Valuation,
)
from finagents.dataflows.screener_csv import parse_screener_csv
from finagents.dataflows.screener_html import parse_screener_html
from finagents.dataflows.session_manager import get_session_manager

CSV_URL_TEMPLATE = "https://www.screener.in/company/{ticker}/consolidated/export/"
HTML_URL_TEMPLATE = "https://www.screener.in/company/{ticker}/consolidated/"

SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.screener.in/",
}


def _is_complete(report: FundamentalsReport) -> bool:
    has_valuation = any(
        v is not None
        for v in [
            report.valuation.pe_ratio,
            report.valuation.pb_ratio,
            report.valuation.market_cap_cr,
            report.valuation.dividend_yield,
        ]
    )
    has_profitability = any(
        v is not None
        for v in [
            report.profitability.roce,
            report.profitability.roe,
        ]
    )
    has_debt = any(
        v is not None
        for v in [
            report.debt_balance.debt_to_equity,
            report.debt_balance.current_ratio,
        ]
    )
    has_shareholding = any(
        v is not None
        for v in [
            report.shareholding.promoter_holding_pct,
            report.shareholding.fii_holding_pct,
            report.shareholding.dii_holding_pct,
        ]
    )
    return has_valuation and has_profitability and has_debt and has_shareholding


def _report_to_bytes(result: DataResult[FundamentalsReport]) -> bytes:
    payload = {
        "status": result.status,
        "source": result.source,
        "error": result.error,
        "data": result.data.model_dump() if result.data else None,
    }
    return json.dumps(payload).encode("utf-8")


def _bytes_to_result(raw: bytes, stale: bool) -> DataResult[FundamentalsReport]:
    payload = json.loads(raw.decode("utf-8"))
    report = None
    if payload.get("data"):
        report = FundamentalsReport(**payload["data"])
    return DataResult(
        status=payload.get("status", "ok"),
        source="cache",
        error=payload.get("error"),
        data=report,
        stale=stale,
    )


async def get_fundamentals(  # noqa: PLR0915
    ticker: str,
    settings: Settings,
) -> DataResult[FundamentalsReport]:
    try:
        safe_ticker = _sanitize_ticker(ticker)
    except ValueError as exc:
        return DataResult(status="error", source="", error=str(exc))

    cache_key = f"fundamentals:{safe_ticker}"
    db_path = cache.CACHE_DB_PATH

    cached = await cache.get_cached("fundamentals", cache_key, db_path=db_path)
    if cached is not None:
        raw, stale = cached
        return _bytes_to_result(raw, stale)

    errors: list[str] = []

    try:
        await rate_limits.acquire("screener.in", settings)
        csv_url = CSV_URL_TEMPLATE.format(ticker=safe_ticker)
        async with httpx.AsyncClient(headers=SCREENER_HEADERS, follow_redirects=True) as client:
            response = await client.get(csv_url)
        content_type = response.headers.get("content-type", "")

        if "text/csv" in content_type:
            report = parse_screener_csv(response.content, safe_ticker)
            status = "ok" if _is_complete(report) else "partial"
            result = DataResult(status=status, source="screener_csv", data=report)
            await cache.set_cached(
                "fundamentals", cache_key, _report_to_bytes(result), db_path=db_path
            )
            return result

        elif "text/html" in content_type:
            errors.append("CSV endpoint returned HTML (login redirect)")

        else:
            errors.append(f"CSV endpoint returned unexpected content-type: {content_type!r}")

    except Exception as exc:
        errors.append(f"Screener CSV failed: {redact_secrets(str(exc))}")

    try:
        await rate_limits.acquire("screener.in", settings)
        html_url = HTML_URL_TEMPLATE.format(ticker=safe_ticker)
        async with httpx.AsyncClient(headers=SCREENER_HEADERS, follow_redirects=True) as client:
            response = await client.get(html_url)
        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type or not content_type:
            report = parse_screener_html(response.content, safe_ticker)
            status = "ok" if _is_complete(report) else "partial"
            result = DataResult(status=status, source="screener_html", data=report)
            await cache.set_cached(
                "fundamentals", cache_key, _report_to_bytes(result), db_path=db_path
            )
            return result

        errors.append(f"HTML endpoint returned unexpected content-type: {content_type!r}")

    except Exception as exc:
        errors.append(f"Screener HTML failed: {redact_secrets(str(exc))}")

    try:
        sm = get_session_manager(settings)
        response_data = await sm.call_with_retry("getFundamentals", safe_ticker)
        data = response_data.get("data", {}) if isinstance(response_data, dict) else {}
        has_any_data = any(
            [
                data.get("pe"),
                data.get("pb"),
                data.get("marketCap"),
                data.get("dividendYield"),
            ]
        )
        if not has_any_data:
            errors.append("Angel One fundamentals returned empty payload")
        else:
            report = FundamentalsReport(
                ticker=safe_ticker,
                valuation=Valuation(
                    pe_ratio=data.get("pe"),
                    pb_ratio=data.get("pb"),
                    market_cap_cr=data.get("marketCap"),
                    dividend_yield=data.get("dividendYield"),
                ),
                profitability=Profitability(),
                debt_balance=DebtBalance(),
                shareholding=Shareholding(),
            )
            result = DataResult(status="partial", source="angelone", data=report)
            await cache.set_cached(
                "fundamentals", cache_key, _report_to_bytes(result), db_path=db_path
            )
            return result

    except Exception as exc:
        errors.append(f"Angel One fundamentals failed: {redact_secrets(str(exc))}")

    return DataResult(
        status="error",
        source="",
        error="; ".join(errors) if errors else "All fundamentals sources failed",
    )
