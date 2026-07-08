from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantsim.engine.metrics import (
    equity_curve_to_series,
    max_drawdown,
    performance_summary,
    returns_from_equity_curve,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)


def make_equity_curve(values):
    base = datetime(2024, 1, 1)
    return [(base + timedelta(days=i), v) for i, v in enumerate(values)]


def test_equity_curve_to_series_empty_input_returns_empty_series():
    series = equity_curve_to_series([])
    assert series.empty


def test_equity_curve_to_series_preserves_order_and_values():
    curve = make_equity_curve([100.0, 110.0, 105.0])
    series = equity_curve_to_series(curve)
    assert list(series.values) == [100.0, 110.0, 105.0]
    assert series.index[0] == curve[0][0]


def test_sharpe_ratio_is_zero_for_empty_returns():
    assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_sharpe_ratio_is_zero_when_returns_have_no_volatility():
    constant_returns = pd.Series([0.01, 0.01, 0.01])
    assert sharpe_ratio(constant_returns) == 0.0


def test_sharpe_ratio_matches_hand_computed_formula():
    returns = pd.Series([0.02, -0.01, 0.015, 0.005])
    expected = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sortino_ratio_is_zero_when_no_downside_returns():
    all_positive = pd.Series([0.01, 0.02, 0.03])
    assert sortino_ratio(all_positive) == 0.0


def test_sortino_ratio_only_penalizes_downside_volatility():
    returns = pd.Series([0.05, -0.02, 0.03, -0.01])
    downside = returns[returns < 0]
    expected = np.sqrt(252) * returns.mean() / downside.std(ddof=0)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_max_drawdown_on_known_equity_path():
    equity = equity_curve_to_series(make_equity_curve([100.0, 120.0, 90.0, 110.0]))
    assert max_drawdown(equity) == pytest.approx((90.0 - 120.0) / 120.0)


def test_max_drawdown_is_zero_for_monotonically_increasing_equity():
    equity = equity_curve_to_series(make_equity_curve([100.0, 110.0, 120.0]))
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_win_rate_counts_positive_pnl_trades():
    assert win_rate([10.0, -5.0, 20.0, -1.0]) == pytest.approx(0.5)


def test_win_rate_is_zero_with_no_trades():
    assert win_rate([]) == 0.0


def test_turnover_divides_total_notional_by_average_equity():
    assert turnover([1000.0, -500.0], average_equity=5000.0) == pytest.approx(0.3)


def test_turnover_is_zero_when_average_equity_is_zero():
    assert turnover([100.0], average_equity=0.0) == 0.0


def test_performance_summary_returns_all_expected_keys():
    curve = make_equity_curve([100.0, 105.0, 102.0, 108.0])
    summary = performance_summary(curve, trade_pnls=[5.0, -2.0], trade_notionals=[1000.0])

    assert set(summary.keys()) == {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "turnover",
    }
    assert summary["win_rate"] == pytest.approx(0.5)


def test_returns_from_equity_curve_computes_pct_change():
    equity = pd.Series([100.0, 110.0, 99.0])
    returns = returns_from_equity_curve(equity)
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)
