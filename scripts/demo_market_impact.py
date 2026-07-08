"""Phase 3 demo: calibrate the square-root market impact model against
simulated order-book flow, plot the cost-vs-size curve, and decompose one
large order's slippage into spread / impact / timing components.

Run with: python scripts/demo_market_impact.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from quantsim.exchange.market_impact import attribute_slippage, calibrate_y_coefficient, square_root_impact
from quantsim.exchange.order_book import OrderBook

T0 = datetime(2024, 1, 1, 9, 30)
MID_PRICE = 100.0
LEVELS = 20
LEVEL_SPACING = 0.01
LEVEL_SIZE = 1_000.0
ASSUMED_ADV = 1_000_000.0
ASSUMED_DAILY_VOL = 0.02
ORDER_SIZES = [50, 200, 500, 1_000, 2_500, 5_000, 10_000]
WORKED_EXAMPLE_SIZE = 5_000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def build_reference_book() -> OrderBook:
    """A fresh order book with `LEVELS` levels of `LEVEL_SIZE` shares on each
    side, spaced `LEVEL_SPACING` apart around `MID_PRICE` — a controlled,
    reproducible book shape for measuring cost-vs-size in isolation from the
    tick generator's own price evolution."""
    book = OrderBook()
    for level_index in range(1, LEVELS + 1):
        book.add_limit("BUY", round(MID_PRICE - level_index * LEVEL_SPACING, 2), LEVEL_SIZE, T0)
        book.add_limit("SELL", round(MID_PRICE + level_index * LEVEL_SPACING, 2), LEVEL_SIZE, T0)
    return book


def measure_fractional_impact(order_size: float) -> float:
    """Submit a fresh market buy of `order_size` shares against a fresh
    reference book; return the realized fractional price impact vs the best
    ask before the order arrived."""
    book = build_reference_book()
    best_ask_before = book.best_ask()
    trades, unfilled = book.market_order("BUY", order_size, T0)
    filled_qty = sum(trade.quantity for trade in trades)
    notional = sum(trade.price * trade.quantity for trade in trades)
    avg_price = notional / filled_qty
    return (avg_price - best_ask_before) / MID_PRICE


def main() -> None:
    observed_impacts = [measure_fractional_impact(size) for size in ORDER_SIZES]
    calibrated_y = calibrate_y_coefficient(observed_impacts, ORDER_SIZES, ASSUMED_ADV, ASSUMED_DAILY_VOL)
    fitted_impacts = [square_root_impact(size, ASSUMED_ADV, ASSUMED_DAILY_VOL, calibrated_y) for size in ORDER_SIZES]

    print("--- Cost-vs-size curve: order-book walk vs calibrated square-root model ---")
    print(f"Calibrated Y coefficient: {calibrated_y:.4f}\n")
    print(f"{'size':>8} {'observed_impact':>16} {'fitted_impact':>16}")
    for size, observed, fitted in zip(ORDER_SIZES, observed_impacts, fitted_impacts):
        print(f"{size:>8} {observed:>16.6f} {fitted:>16.6f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ORDER_SIZES, observed_impacts, "o-", label="Observed (order-book walk)")
    plt.plot(ORDER_SIZES, fitted_impacts, "--", label=f"Fitted sqrt law (Y={calibrated_y:.3f})")
    plt.xlabel("Order size (shares)")
    plt.ylabel("Fractional price impact")
    plt.title("Market impact: cost vs. order size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_impact_cost_curve.png")

    print("\n--- Worked example: decomposing one large order's slippage ---")
    book = build_reference_book()
    best_ask_at_arrival = book.best_ask()
    trades, unfilled = book.market_order("BUY", WORKED_EXAMPLE_SIZE, T0)
    filled_qty = sum(trade.quantity for trade in trades)
    notional = sum(trade.price * trade.quantity for trade in trades)
    avg_fill_price = notional / filled_qty

    attribution = attribute_slippage(
        fill_price=avg_fill_price,
        quantity=filled_qty,
        direction="BUY",
        mid_price_at_decision=MID_PRICE,
        mid_price_at_arrival=MID_PRICE,  # no timing delay modeled in this synchronous example
        best_price_at_arrival=best_ask_at_arrival,
    )

    print(f"Order size: {WORKED_EXAMPLE_SIZE} shares, filled: {filled_qty:.0f} shares, unfilled: {unfilled:.0f}")
    print(f"Avg fill price: {avg_fill_price:.4f} (mid at decision: {MID_PRICE:.4f})")
    print(f"  Timing cost:  ${attribution.timing_cost:9.2f}")
    print(f"  Spread cost:  ${attribution.spread_cost:9.2f}")
    print(f"  Impact cost:  ${attribution.impact_cost:9.2f}")
    print(f"  Total cost:   ${attribution.total:9.2f}")
    print(f"\nSaved cost-vs-size plot to {OUTPUT_DIR / 'market_impact_cost_curve.png'}")


if __name__ == "__main__":
    main()
