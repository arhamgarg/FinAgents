from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Valuation(BaseModel):
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    market_cap_cr: Optional[float] = None
    dividend_yield: Optional[float] = None


class Profitability(BaseModel):
    roce: Optional[float] = None
    roe: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None


class DebtBalance(BaseModel):
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    current_ratio: Optional[float] = None


class QuarterRow(BaseModel):
    quarter: str
    revenue_cr: Optional[float] = None
    pat_cr: Optional[float] = None


class Shareholding(BaseModel):
    promoter_holding_pct: Optional[float] = None
    promoter_pledge_pct: Optional[float] = None
    fii_holding_pct: Optional[float] = None
    dii_holding_pct: Optional[float] = None


class FundamentalsReport(BaseModel):
    ticker: str
    valuation: Valuation
    profitability: Profitability
    debt_balance: DebtBalance
    shareholding: Shareholding
    quarterly_trend: list[QuarterRow] = []
