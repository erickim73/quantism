"""Phase 1 demo: moving-average crossover backtest on daily AAPL data.

Run with: python scripts/demo_ma_crossover.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from quantsim.data.loaders import HistoricDataHandler, align_frames, load_yfinance_ohlcv
from quantsim.engine.backtester import Backtester
from quantsim.engine.event_queue import EventQueue
from quantsim.engine.execution import SimpleExecutionHandler
from quantsim.engine.portfolio import Portfolio
from quantsim.strategies.momentum import MovingAverageCrossoverStrategy

SYMBOL = "AAPL"
START = "2023-01-01"
END = "2024-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    frames = align_frames(load_yfinance_ohlcv([SYMBOL], START, END))
    queue = EventQueue()
    data_handler = HistoricDataHandler(frames, queue)
    strategy = MovingAverageCrossoverStrategy(symbols=[SYMBOL], short_window=10, long_window=30)
    execution_handler = SimpleExecutionHandler(data_handler, commission_bps=1.0, slippage_bps=1.0)
    portfolio = Portfolio(initial_cash=100_000.0, symbols=[SYMBOL])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=100)
    summary = backtester.run()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"--- {SYMBOL} MA Crossover ({START} to {END}) ---")
    if portfolio.equity_curve:
        print(f"Final equity: {portfolio.equity_curve[-1][1]:,.2f}")
    print(f"Total fills: {len(backtester.fills)}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    if portfolio.equity_curve:
        dates, equity = zip(*portfolio.equity_curve)
        plt.figure(figsize=(10, 5))
        plt.plot(dates, equity)
        plt.title(f"{SYMBOL} MA Crossover Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "equity_curve.png")

    with open(OUTPUT_DIR / "trade_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "symbol", "direction", "quantity", "fill_price", "commission", "slippage"]
        )
        for fill in backtester.fills:
            writer.writerow(
                [fill.timestamp, fill.symbol, fill.direction, fill.quantity, fill.fill_price, fill.commission, fill.slippage]
            )

    print(f"Saved equity curve plot and trade log to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
