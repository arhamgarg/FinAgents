from __future__ import annotations

import io
import json
from typing import Optional

import httpx

from financialagents.config import Settings
from financialagents.dataflows import cache, rate_limits
from financialagents.dataflows._ticker import sanitize_ticker as _sanitize_ticker
from financialagents.dataflows.data_result import DataResult
from financialagents.dataflows.fundamentals_schema import (
    DebtBalance,
    FundamentalsReport,
    Profitability,
    QuarterRow,
    Shareholding,
    Valuation,
)
from financialagents.dataflows.session_manager import get_session_manager


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



def _parse_screener_csv(csv_bytes: bytes, ticker: str) -> FundamentalsReport:
    import csv

    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))

    kv: dict[str, str] = {}
    quarters: list[str] = []
    sales_row: list[str] = []
    profit_row: list[str] = []
    in_profit_loss = False

    for row in reader:
        if not row:
            in_profit_loss = False
            continue
        label = row[0].strip()
        if label.upper() == "PROFIT & LOSS":
            in_profit_loss = True
            continue
        if in_profit_loss:
            if label.lower() == "quarter":
                quarters = [c.strip() for c in row[1:]]
            elif label.lower() in ("sales", "revenue"):
                sales_row = [c.strip() for c in row[1:]]
            elif label.lower() in ("net profit", "pat"):
                profit_row = [c.strip() for c in row[1:]]
            continue
        if len(row) >= 2 and row[1].strip():
            kv[label] = row[1].strip()

    def _f(key: str) -> Optional[float]:
        val = kv.get(key, "").replace(",", "").strip()
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    valuation = Valuation(
        pe_ratio=_f("Price to Earning"),
        pb_ratio=_f("Price to book value"),
        market_cap_cr=_f("Market Capitalization"),
        dividend_yield=_f("Dividend Yield"),
    )

    profitability = Profitability(
        roce=_f("Return on capital employed"),
        roe=_f("Return on equity"),
        operating_margin=_f("OPM"),
    )

    debt_balance = DebtBalance(
        debt_to_equity=_f("Debt to equity"),
        interest_coverage=_f("Interest Coverage Ratio"),
        current_ratio=_f("Current Ratio"),
    )

    promoter = _f("Promoter holding")
    pledge = _f("Pledged percentage")
    fii = _f("FII holding")
    dii = _f("DII holding")

    shareholding = Shareholding(
        promoter_holding_pct=promoter,
        promoter_pledge_pct=pledge,
        fii_holding_pct=fii,
        dii_holding_pct=dii,
    )

    quarterly_trend: list[QuarterRow] = []
    if quarters and (sales_row or profit_row):
        cols = min(len(quarters), 4)
        for i in range(cols):
            rev = None
            pat = None
            if i < len(sales_row):
                try:
                    rev = float(sales_row[i].replace(",", ""))
                except ValueError:
                    pass
            if i < len(profit_row):
                try:
                    pat = float(profit_row[i].replace(",", ""))
                except ValueError:
                    pass
            quarterly_trend.append(
                QuarterRow(quarter=quarters[i], revenue_cr=rev, pat_cr=pat)
            )

    return FundamentalsReport(
        ticker=ticker,
        valuation=valuation,
        profitability=profitability,
        debt_balance=debt_balance,
        shareholding=shareholding,
        quarterly_trend=quarterly_trend,
    )



def _parse_screener_html(html_bytes: bytes, ticker: str) -> FundamentalsReport:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_bytes, "lxml")

    def _extract_float(text: str) -> Optional[float]:
        clean = text.replace("%", "").replace(",", "").strip()
        if not clean:
            return None
        try:
            return float(clean)
        except ValueError:
            return None

    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    market_cap_cr: Optional[float] = None
    dividend_yield: Optional[float] = None
    roce: Optional[float] = None
    roe: Optional[float] = None

    top_ratios = soup.find("section", id="top-ratios")
    if top_ratios:
        for li in top_ratios.find_all("li"):
            name_el = li.find(class_="name")
            num_el = li.find(class_="number")
            if not name_el or not num_el:
                continue
            name = name_el.get_text(strip=True).lower()
            val = _extract_float(num_el.get_text(strip=True))
            if "p/e" in name or "stock p/e" in name or name == "stock p/e":
                pe_ratio = val
            elif "price to book" in name or "p/b" in name:
                pb_ratio = val
            elif "market cap" in name:
                market_cap_cr = val
            elif "dividend yield" in name:
                dividend_yield = val
            elif name == "roce" or "return on capital" in name:
                roce = val
            elif name == "roe" or "return on equity" in name:
                roe = val

    valuation = Valuation(
        pe_ratio=pe_ratio,
        pb_ratio=pb_ratio,
        market_cap_cr=market_cap_cr,
        dividend_yield=dividend_yield,
    )

    profitability = Profitability(roce=roce, roe=roe)

    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None

    balance_sheet = soup.find("section", id="balance-sheet")
    if balance_sheet:
        for tr in balance_sheet.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                val_text = cells[1].get_text(strip=True)
                val = _extract_float(val_text)
                if "debt to equity" in label or "debt/equity" in label:
                    debt_to_equity = val
                elif "current ratio" in label:
                    current_ratio = val

    debt_balance = DebtBalance(
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
    )

    promoter_pct: Optional[float] = None
    pledge_pct: Optional[float] = None
    fii_pct: Optional[float] = None
    dii_pct: Optional[float] = None

    shareholding_section = soup.find("section", id="shareholding")
    if shareholding_section:
        for tr in shareholding_section.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                val = _extract_float(cells[1].get_text(strip=True))
                if "promoter" in label:
                    promoter_pct = val
                elif "fii" in label:
                    fii_pct = val
                elif "dii" in label:
                    dii_pct = val
                elif "pledge" in label:
                    pledge_pct = val

    shareholding = Shareholding(
        promoter_holding_pct=promoter_pct,
        promoter_pledge_pct=pledge_pct,
        fii_holding_pct=fii_pct,
        dii_holding_pct=dii_pct,
    )

    quarterly_trend: list[QuarterRow] = []
    profit_loss = soup.find("section", id="profit-loss")
    if profit_loss:
        table = profit_loss.find("table")
        if table:
            rows = table.find_all("tr")
            if rows:
                header_row = rows[0]
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
                quarter_headers = headers[1:5]

                sales_vals: list[Optional[float]] = []
                pat_vals: list[Optional[float]] = []

                for tr in rows[1:]:
                    cells = tr.find_all(["th", "td"])
                    if not cells:
                        continue
                    label = cells[0].get_text(strip=True).lower()
                    data = [_extract_float(c.get_text(strip=True)) for c in cells[1:5]]
                    if label in ("sales", "revenue"):
                        sales_vals = data
                    elif "net profit" in label or "pat" in label:
                        pat_vals = data

                for i, q in enumerate(quarter_headers):
                    rev = sales_vals[i] if i < len(sales_vals) else None
                    pat = pat_vals[i] if i < len(pat_vals) else None
                    quarterly_trend.append(
                        QuarterRow(quarter=q, revenue_cr=rev, pat_cr=pat)
                    )

    return FundamentalsReport(
        ticker=ticker,
        valuation=valuation,
        profitability=profitability,
        debt_balance=debt_balance,
        shareholding=shareholding,
        quarterly_trend=quarterly_trend,
    )



def _is_complete(report: FundamentalsReport) -> bool:
    has_valuation = any(v is not None for v in [
        report.valuation.pe_ratio,
        report.valuation.pb_ratio,
        report.valuation.market_cap_cr,
        report.valuation.dividend_yield,
    ])
    has_profitability = any(v is not None for v in [
        report.profitability.roce,
        report.profitability.roe,
    ])
    has_debt = any(v is not None for v in [
        report.debt_balance.debt_to_equity,
        report.debt_balance.current_ratio,
    ])
    has_shareholding = any(v is not None for v in [
        report.shareholding.promoter_holding_pct,
        report.shareholding.fii_holding_pct,
        report.shareholding.dii_holding_pct,
    ])
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



async def get_fundamentals(
    ticker: str,
    settings: Optional[Settings] = None,
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
        await rate_limits.acquire("screener.in")
        csv_url = CSV_URL_TEMPLATE.format(ticker=safe_ticker)
        async with httpx.AsyncClient(headers=SCREENER_HEADERS, follow_redirects=True) as client:
            response = await client.get(csv_url)
        content_type = response.headers.get("content-type", "")

        if "text/csv" in content_type:
            report = _parse_screener_csv(response.content, safe_ticker)
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

    except Exception as exc:  # noqa: BLE001
        errors.append(f"Screener CSV failed: {exc}")

    try:
        await rate_limits.acquire("screener.in")
        html_url = HTML_URL_TEMPLATE.format(ticker=safe_ticker)
        async with httpx.AsyncClient(headers=SCREENER_HEADERS, follow_redirects=True) as client:
            response = await client.get(html_url)
        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type or not content_type:
            report = _parse_screener_html(response.content, safe_ticker)
            status = "ok" if _is_complete(report) else "partial"
            result = DataResult(status=status, source="screener_html", data=report)
            await cache.set_cached(
                "fundamentals", cache_key, _report_to_bytes(result), db_path=db_path
            )
            return result

        errors.append(f"HTML endpoint returned unexpected content-type: {content_type!r}")

    except Exception as exc:  # noqa: BLE001
        errors.append(f"Screener HTML failed: {exc}")

    if settings is None:
        errors.append(
            "Angel One fundamentals requires settings; none provided "
            "(tier-3 skipped to avoid poisoning session manager singleton)"
        )
        return DataResult(
            status="error",
            source="",
            error="; ".join(errors),
        )
    try:
        sm = get_session_manager(settings)
        response_data = await sm.call_with_retry(
            "getFundamentals", safe_ticker
        )
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

    except Exception as exc:  # noqa: BLE001
        errors.append(f"Angel One fundamentals failed: {exc}")

    return DataResult(
        status="error",
        source="",
        error="; ".join(errors) if errors else "All fundamentals sources failed",
    )
