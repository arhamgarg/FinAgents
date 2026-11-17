from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Valuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    market_cap_cr: float | None = None
    dividend_yield: float | None = None


class Profitability(BaseModel):
    model_config = ConfigDict(frozen=True)

    roce: float | None = None
    roe: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None


class DebtBalance(BaseModel):
    model_config = ConfigDict(frozen=True)

    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    current_ratio: float | None = None


class QuarterRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    quarter: str
    revenue_cr: float | None = None
    pat_cr: float | None = None


class Shareholding(BaseModel):
    model_config = ConfigDict(frozen=True)

    promoter_holding_pct: float | None = None
    promoter_pledge_pct: float | None = None
    fii_holding_pct: float | None = None
    dii_holding_pct: float | None = None


class FundamentalsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    valuation: Valuation
    profitability: Profitability
    debt_balance: DebtBalance
    shareholding: Shareholding
    quarterly_trend: list[QuarterRow] = []
