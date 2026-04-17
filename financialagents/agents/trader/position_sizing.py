from __future__ import annotations

import math

POSITION_SIZE_HARD_CAP = 0.05


def atr_based_size(
    capital: float,
    risk_per_trade_pct: float,
    atr: float,
    atr_multiplier: float,
    entry_price: float,
) -> tuple[float, int]:
    stop_distance = atr * atr_multiplier
    if stop_distance <= 0 or entry_price <= 0 or capital <= 0:
        return 0.0, 0
    risk_amount = capital * risk_per_trade_pct
    shares = math.floor(risk_amount / stop_distance)
    if shares <= 0:
        return 0.0, 0
    pct = (shares * entry_price) / capital
    return pct, shares


def kelly_size(
    win_rate: float,
    reward_ratio: float,
    kelly_fraction: float,
) -> float:
    if reward_ratio <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    full_kelly = (reward_ratio * p - q) / reward_ratio
    if full_kelly <= 0:
        return 0.0
    scaled = kelly_fraction * full_kelly
    return min(scaled, POSITION_SIZE_HARD_CAP)


def fixed_fractional_size(
    capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
) -> tuple[float, int]:
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0 or entry_price <= 0 or capital <= 0:
        return 0.0, 0
    risk_amount = capital * risk_per_trade_pct
    shares = math.floor(risk_amount / stop_distance)
    if shares <= 0:
        return 0.0, 0
    pct = (shares * entry_price) / capital
    return pct, shares
