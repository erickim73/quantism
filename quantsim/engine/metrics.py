from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def equity_curve_to_series(equity_curve: list[tuple[datetime, float]]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    timestamps, values = zip(*equity_curve)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps))


def returns_from_equity_curve(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if downside.empty or downside_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)


def turnover(trade_notionals: list[float], average_equity: float) -> float:
    if average_equity == 0:
        return 0.0
    return sum(abs(v) for v in trade_notionals) / average_equity


def performance_summary(
    equity_curve: list[tuple[datetime, float]],
    trade_pnls: list[float],
    trade_notionals: list[float],
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    equity = equity_curve_to_series(equity_curve)
    returns = returns_from_equity_curve(equity)
    average_equity = float(equity.mean()) if not equity.empty else 0.0

    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trade_pnls),
        "turnover": turnover(trade_notionals, average_equity),
    }
