"""Phase 5 demo: cointegration-based pairs trading on two synthetically
generated, genuinely cointegrated instruments (a common stochastic trend plus
idiosyncratic noise per leg — this sandbox has no network access to fetch a
real correlated pair).

Run with: python scripts/demo_pairs_trading.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantsim.data.loaders import HistoricDataHandler, align_frames
from quantsim.engine.backtester import Backtester
from quantsim.engine.event_queue import EventQueue
from quantsim.engine.execution import SimpleExecutionHandler
from quantsim.engine.metrics import performance_summary
from quantsim.engine.portfolio import Portfolio
from quantsim.strategies.pairs_trading import PairsTradingStrategy, is_cointegrated

SEED = 21
N_DAYS = 400
SYMBOL_A = "PAIR_A"
SYMBOL_B = "PAIR_B"


def build_cointegrated_pair() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2023-01-02", periods=N_DAYS)

    common_trend = np.cumsum(rng.normal(0.0002, 0.01, N_DAYS))
    price_a = 100 * np.exp(common_trend + rng.normal(0, 0.005, N_DAYS))
    price_b = 50 * np.exp(0.6 * common_trend + rng.normal(0, 0.005, N_DAYS))

    frames = {}
    for symbol, price in ((SYMBOL_A, price_a), (SYMBOL_B, price_b)):
        frames[symbol] = pd.DataFrame(
            {"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 100_000},
            index=dates,
        )
    return frames


def main() -> None:
    frames = align_frames(build_cointegrated_pair())

    coint_check = is_cointegrated(frames[SYMBOL_A]["close"].to_numpy(), frames[SYMBOL_B]["close"].to_numpy())
    print(f"Engle-Granger cointegration test (full history): cointegrated={coint_check}\n")

    data_handler = HistoricDataHandler(frames, EventQueue())
    strategy = PairsTradingStrategy(SYMBOL_A, SYMBOL_B, window=30, entry_z=2.0, exit_z=0.5)
    execution_handler = SimpleExecutionHandler(data_handler, commission_bps=1.0, slippage_bps=1.0)
    portfolio = Portfolio(initial_cash=100_000.0, symbols=[SYMBOL_A, SYMBOL_B])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=100)
    backtester.run()

    summary = performance_summary(portfolio.equity_curve, backtester.trade_pnls, backtester.trade_notionals)

    print(f"--- Pairs trading: {SYMBOL_A} / {SYMBOL_B} ---")
    print(f"Fills: {len(backtester.fills)}")
    if portfolio.equity_curve:
        print(f"Final equity: {portfolio.equity_curve[-1][1]:,.2f}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
