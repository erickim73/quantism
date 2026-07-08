"""Phase 2 demo: the same MA-crossover strategy replayed against an identical
synthetic tick stream (same seed -> identical mid-price path), comparing the
Phase 1 naive fixed-bps execution model against real order-book execution.

Run with: python scripts/demo_tick_replay.py
"""

from __future__ import annotations

from datetime import datetime

from quantsim.data.loaders import HistoricDataHandler
from quantsim.engine.backtester import Backtester
from quantsim.engine.event_queue import EventQueue
from quantsim.engine.execution import SimpleExecutionHandler
from quantsim.engine.portfolio import Portfolio
from quantsim.exchange.execution import OrderBookExecutionHandler
from quantsim.exchange.tick_generator import SyntheticTickDataHandler, SyntheticTickGenerator
from quantsim.strategies.momentum import MovingAverageCrossoverStrategy

SYMBOL = "SYN"
SEED = 42
N_TICKS = 500
START = datetime(2024, 1, 2, 9, 30)
MEAN_INTERVAL_SECONDS = 2.0
SHORT_WINDOW = 10
LONG_WINDOW = 30
# Deliberately larger than one order-book level (100 shares/level) so orders
# walk multiple levels and the book-depth slippage story is actually visible.
ORDER_QUANTITY = 250


def run_naive_execution() -> tuple[Portfolio, Backtester]:
    """Same synthetic mid-price path (same seed), filled with Phase 1's fixed-
    bps SimpleExecutionHandler at the next tick's open."""
    generator = SyntheticTickGenerator(seed=SEED)
    frames = {SYMBOL: generator.generate_ohlcv(N_TICKS, START, MEAN_INTERVAL_SECONDS)}

    data_handler = HistoricDataHandler(frames, EventQueue())
    strategy = MovingAverageCrossoverStrategy([SYMBOL], SHORT_WINDOW, LONG_WINDOW)
    execution_handler = SimpleExecutionHandler(data_handler, commission_bps=1.0, slippage_bps=1.0)
    portfolio = Portfolio(initial_cash=100_000.0, symbols=[SYMBOL])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=ORDER_QUANTITY)
    backtester.run()
    return portfolio, backtester


def run_order_book_execution() -> tuple[Portfolio, Backtester]:
    """Same seed -> identical mid-price path, but fills come from actually
    walking a live limit order book instead of a fixed bps model."""
    generator = SyntheticTickGenerator(seed=SEED)
    data_handler = SyntheticTickDataHandler(SYMBOL, generator, N_TICKS, START, MEAN_INTERVAL_SECONDS)
    strategy = MovingAverageCrossoverStrategy([SYMBOL], SHORT_WINDOW, LONG_WINDOW)
    execution_handler = OrderBookExecutionHandler(generator.order_book, commission_bps=1.0)
    portfolio = Portfolio(initial_cash=100_000.0, symbols=[SYMBOL])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=ORDER_QUANTITY)
    backtester.run()
    return portfolio, backtester


def summarize(label: str, portfolio: Portfolio, backtester: Backtester) -> None:
    total_slippage = sum(fill.slippage for fill in backtester.fills)
    avg_slippage = total_slippage / len(backtester.fills) if backtester.fills else 0.0
    final_equity = portfolio.equity_curve[-1][1] if portfolio.equity_curve else portfolio.cash

    print(f"--- {label} ---")
    print(f"Fills: {len(backtester.fills)}")
    print(f"Total slippage cost: {total_slippage:.4f}")
    print(f"Avg slippage per fill: {avg_slippage:.4f}")
    print(f"Final equity: {final_equity:,.2f}")
    print()


def main() -> None:
    naive_portfolio, naive_backtester = run_naive_execution()
    book_portfolio, book_backtester = run_order_book_execution()

    summarize("Phase 1 naive execution (fixed bps)", naive_portfolio, naive_backtester)
    summarize("Phase 2 order-book execution (real depth)", book_portfolio, book_backtester)


if __name__ == "__main__":
    main()
