from __future__ import annotations

import contextlib
import csv
import io

from finagents.dataflows.fundamentals_schema import (
    DebtBalance,
    FundamentalsReport,
    Profitability,
    QuarterRow,
    Shareholding,
    Valuation,
)

LABELED_ROW_MIN_CELLS = 2


def parse_screener_csv(csv_bytes: bytes, ticker: str) -> FundamentalsReport:
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
        if len(row) >= LABELED_ROW_MIN_CELLS and row[1].strip():
            kv[label] = row[1].strip()

    def _f(key: str) -> float | None:
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
                with contextlib.suppress(ValueError):
                    rev = float(sales_row[i].replace(",", ""))
            if i < len(profit_row):
                with contextlib.suppress(ValueError):
                    pat = float(profit_row[i].replace(",", ""))
            quarterly_trend.append(QuarterRow(quarter=quarters[i], revenue_cr=rev, pat_cr=pat))

    return FundamentalsReport(
        ticker=ticker,
        valuation=valuation,
        profitability=profitability,
        debt_balance=debt_balance,
        shareholding=shareholding,
        quarterly_trend=quarterly_trend,
    )
