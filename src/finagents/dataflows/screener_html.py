from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from finagents.dataflows.fundamentals_schema import (
    DebtBalance,
    FundamentalsReport,
    Profitability,
    QuarterRow,
    Shareholding,
    Valuation,
)

LABELED_ROW_MIN_CELLS = 2
RECENT_QUARTERS = 4


def parse_screener_html(html_bytes: bytes, ticker: str) -> FundamentalsReport:  # noqa: PLR0912, PLR0915
    soup = BeautifulSoup(html_bytes, "lxml")

    def _extract_float(text: str) -> float | None:
        clean = text.replace("₹", "").replace("%", "").replace(",", "").replace("\xa0", "").strip()
        if not clean:
            return None
        try:
            return float(clean)
        except ValueError:
            return None

    def _li_name_and_value(li: Tag) -> tuple[str, float | None]:
        name_el = li.find(class_="name")
        num_el = li.find(class_="number")
        if name_el and num_el:
            return name_el.get_text(strip=True).lower(), _extract_float(num_el.get_text(strip=True))

        spans = li.find_all("span")
        if len(spans) >= LABELED_ROW_MIN_CELLS:
            name = spans[0].get_text(strip=True).lower()
            val = _extract_float(spans[-1].get_text(strip=True))
            return name, val

        return "", None

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    pb_from_book_value: float | None = None
    current_price: float | None = None
    book_value: float | None = None
    market_cap_cr: float | None = None
    dividend_yield: float | None = None
    roce: float | None = None
    roe: float | None = None

    top_ratios_el = soup.find(id="top-ratios")
    if top_ratios_el:
        for li in top_ratios_el.find_all("li"):
            name, val = _li_name_and_value(li)
            if not name:
                continue
            if "stock p/e" in name or name == "p/e" or (("p/e" in name) and "price to" not in name):
                pe_ratio = val
            elif "price to book" in name or "p/b" in name:
                pb_ratio = val
            elif "book value" in name:
                book_value = val
            elif "market cap" in name:
                market_cap_cr = val
            elif "current price" in name:
                current_price = val
            elif "dividend yield" in name:
                dividend_yield = val
            elif name == "roce" or "return on capital" in name:
                roce = val
            elif name == "roe" or "return on equity" in name:
                roe = val

    if pb_ratio is None and current_price is not None and book_value is not None and book_value > 0:
        pb_from_book_value = round(current_price / book_value, 2)

    valuation = Valuation(
        pe_ratio=pe_ratio,
        pb_ratio=pb_ratio if pb_ratio is not None else pb_from_book_value,
        market_cap_cr=market_cap_cr,
        dividend_yield=dividend_yield,
    )

    profitability = Profitability(roce=roce, roe=roe)

    debt_to_equity: float | None = None
    current_ratio: float | None = None

    balance_sheet = soup.find("section", id="balance-sheet")
    if balance_sheet:
        table = balance_sheet.find("table")
        if table:
            rows = table.find_all("tr")
            header_cells = rows[0].find_all(["th", "td"]) if rows else []
            last_col = len(header_cells) - 1 if len(header_cells) > 1 else 1

            borrowings: float | None = None
            equity_capital: float | None = None
            reserves: float | None = None

            for tr in rows[1:]:
                cells = tr.find_all(["th", "td"])
                if len(cells) < LABELED_ROW_MIN_CELLS:
                    continue
                label = cells[0].get_text(strip=True).lower()
                col_idx = min(last_col, len(cells) - 1)
                val = _extract_float(cells[col_idx].get_text(strip=True))
                if "debt to equity" in label or "debt/equity" in label:
                    debt_to_equity = val
                elif "current ratio" in label:
                    current_ratio = val
                elif label.startswith("borrowings"):
                    borrowings = val
                elif label.startswith("equity capital"):
                    equity_capital = val
                elif label.startswith("reserves"):
                    reserves = val

            if debt_to_equity is None and borrowings is not None:
                total_equity = (equity_capital or 0.0) + (reserves or 0.0)
                if total_equity > 0:
                    debt_to_equity = round(borrowings / total_equity, 2)

    debt_balance = DebtBalance(
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
    )

    promoter_pct: float | None = None
    pledge_pct: float | None = None
    fii_pct: float | None = None
    dii_pct: float | None = None

    shareholding_section = soup.find("section", id="shareholding")
    if shareholding_section:
        table = shareholding_section.find("table")
        if table:
            rows = table.find_all("tr")
            header_cells = rows[0].find_all(["th", "td"]) if rows else []
            last_col = len(header_cells) - 1 if len(header_cells) > 1 else 1

            for tr in rows[1:]:
                cells = tr.find_all(["th", "td"])
                if len(cells) < LABELED_ROW_MIN_CELLS:
                    continue
                label = cells[0].get_text(strip=True).lower()
                col_idx = min(last_col, len(cells) - 1)
                val = _extract_float(cells[col_idx].get_text(strip=True))
                if "promoter" in label:
                    promoter_pct = val
                elif "fii" in label:
                    fii_pct = val
                elif "dii" in label:
                    dii_pct = val
                elif "pledge" in label:
                    pledge_pct = val
        else:
            for tr in shareholding_section.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if len(cells) >= LABELED_ROW_MIN_CELLS:
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

    def _parse_pl_table(section_el: Tag | None) -> list[QuarterRow]:
        table = section_el.find("table") if section_el else None
        if not table:
            return []
        rows = table.find_all("tr")
        if not rows:
            return []

        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        data_headers = headers[1:]
        recent_headers = (
            data_headers[-RECENT_QUARTERS:]
            if len(data_headers) >= RECENT_QUARTERS
            else data_headers
        )
        n = len(recent_headers)
        col_start = len(headers) - n

        sales_vals: list[float | None] = [None] * n
        pat_vals: list[float | None] = [None] * n

        for tr in rows[1:]:
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            label = cells[0].get_text(strip=True).lower()
            label_clean = label.rstrip("+").strip()
            data = [
                _extract_float(cells[col_start + i].get_text(strip=True))
                if col_start + i < len(cells)
                else None
                for i in range(n)
            ]
            if label_clean in ("sales", "revenue"):
                sales_vals = data
            elif "net profit" in label_clean or label_clean == "pat":
                pat_vals = data

        result: list[QuarterRow] = []
        for i, q in enumerate(recent_headers):
            result.append(
                QuarterRow(
                    quarter=q,
                    revenue_cr=sales_vals[i] if i < len(sales_vals) else None,
                    pat_cr=pat_vals[i] if i < len(pat_vals) else None,
                )
            )
        return result

    quarters_section = soup.find("section", id="quarters")
    if quarters_section:
        quarterly_trend = _parse_pl_table(quarters_section)

    if not quarterly_trend:
        profit_loss = soup.find("section", id="profit-loss")
        if profit_loss:
            quarterly_trend = _parse_pl_table(profit_loss)

    return FundamentalsReport(
        ticker=ticker,
        valuation=valuation,
        profitability=profitability,
        debt_balance=debt_balance,
        shareholding=shareholding,
        quarterly_trend=quarterly_trend,
    )
