"""Square-root market impact model and slippage attribution.

The square-root law is a well-documented empirical regularity: the price
impact of a metaorder scales with the square root of its size relative to
average daily volume (ADV), I(Q) = Y * sigma * sqrt(Q / ADV), where sigma is
daily return volatility and Y is an empirical prefactor typically calibrated
in [0.5, 1.0] for US equities (see e.g. Almgren, Thum, Hauptmann & Li (2005),
"Direct Estimation of Equity Market Impact").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantsim.engine.event_queue import FillEvent, OrderEvent
from quantsim.engine.execution import ExecutionHandler
from quantsim.exchange.tick_generator import SyntheticTickGenerator

DEFAULT_Y_COEFFICIENT = 0.75


def square_root_impact(
    order_size: float,
    average_daily_volume: float,
    daily_volatility: float,
    y_coefficient: float = DEFAULT_Y_COEFFICIENT,
) -> float:
    """Expected *fractional* price impact of a metaorder: I(Q) = Y * sigma *
    sqrt(Q / ADV). Multiply by price to get a dollar-per-share impact."""
    if average_daily_volume <= 0:
        raise ValueError("average_daily_volume must be positive")
    participation = abs(order_size) / average_daily_volume
    return y_coefficient * daily_volatility * participation**0.5


def calibrate_y_coefficient(
    observed_fractional_impacts: list[float],
    order_sizes: list[float],
    average_daily_volume: float,
    daily_volatility: float,
) -> float:
    """Least-squares calibration of Y from observed fractional impacts against
    simulated order flow, holding ADV/volatility fixed. Closed-form OLS
    solution for a single coefficient on a no-intercept regressor."""
    sizes = np.asarray(order_sizes, dtype=float)
    observed = np.asarray(observed_fractional_impacts, dtype=float)
    predictor = daily_volatility * np.sqrt(np.abs(sizes) / average_daily_volume)

    denominator = float(np.sum(predictor**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(predictor * observed) / denominator)


@dataclass(frozen=True)
class SlippageAttribution:
    """Decomposition of realized trading cost (in dollars, positive = cost to
    the trader) into three components, relative to the mid-price at decision
    time."""

    spread_cost: float
    impact_cost: float
    timing_cost: float

    @property
    def total(self) -> float:
        return self.spread_cost + self.impact_cost + self.timing_cost


def attribute_slippage(
    fill_price: float,
    quantity: float,
    direction: str,
    mid_price_at_decision: float,
    mid_price_at_arrival: float,
    best_price_at_arrival: float,
) -> SlippageAttribution:
    """Decompose realized slippage vs the mid-price at decision time into:

    - timing_cost: price drift between the decision and the order's arrival
      (e.g. the bar/tick delay before a naive execution handler fills it)
    - spread_cost: cost of crossing from mid to the touch (best bid/ask) at
      arrival
    - impact_cost: cost of walking beyond the touch to get fully filled

    All costs are signed so that a positive value is a cost to the trader.
    """
    direction_sign = 1 if direction == "BUY" else -1

    timing_cost = direction_sign * (mid_price_at_arrival - mid_price_at_decision) * quantity
    spread_cost = direction_sign * (best_price_at_arrival - mid_price_at_arrival) * quantity
    impact_cost = direction_sign * (fill_price - best_price_at_arrival) * quantity

    return SlippageAttribution(spread_cost=spread_cost, impact_cost=impact_cost, timing_cost=timing_cost)


class MarketImpactExecutionHandler(ExecutionHandler):
    """Wraps another ExecutionHandler and, after every fill, permanently
    shifts the underlying market's mid-price by the square-root model's
    predicted impact.

    An order book alone only reproduces the *mechanical* cost of walking
    today's resting liquidity (Phase 2). It does not model the fact that a
    large trade also moves the market for subsequent trades — that lasting
    effect is what the square-root law describes, so this handler applies it
    on top of (not instead of) the order book's own walk-the-book slippage.
    """

    def __init__(
        self,
        inner: ExecutionHandler,
        tick_generator: SyntheticTickGenerator,
        average_daily_volume: float,
        daily_volatility: float,
        y_coefficient: float = DEFAULT_Y_COEFFICIENT,
    ) -> None:
        self.inner = inner
        self.tick_generator = tick_generator
        self.average_daily_volume = average_daily_volume
        self.daily_volatility = daily_volatility
        self.y_coefficient = y_coefficient
        self.impact_log: list[float] = []

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        fill = self.inner.execute_order(order)
        if fill is None:
            return None

        fractional_impact = square_root_impact(
            fill.quantity, self.average_daily_volume, self.daily_volatility, self.y_coefficient
        )
        direction_sign = 1 if fill.direction == "BUY" else -1
        price_shift = direction_sign * fractional_impact * self.tick_generator.mid_price

        self.tick_generator.mid_price = max(0.01, self.tick_generator.mid_price + price_shift)
        self.impact_log.append(price_shift)

        return fill
