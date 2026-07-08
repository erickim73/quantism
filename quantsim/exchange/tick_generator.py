from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantsim.exchange.order_book import OrderBook

DEFAULT_LEVELS = 5
DEFAULT_LEVEL_SPACING = 0.01
DEFAULT_LEVEL_SIZE = 100.0


@dataclass(frozen=True)
class SyntheticTick:
    timestamp: datetime
    mid_price: float
    best_bid: float | None
    best_ask: float | None


class SyntheticTickGenerator:
    """Generates a synthetic tick stream to drive an OrderBook without needing
    real exchange tick data (e.g. LOBSTER).

    The midpoint follows a discretized Brownian motion. At each tick, all
    resting liquidity is cancelled and re-quoted at `levels` price levels on
    both sides around the new midpoint, offset by a shared random "bounce"
    (the same draw applied to both sides, so the bid-ask gap is unaffected —
    the book can never end up crossed by construction). Inter-tick arrival
    times are drawn from an exponential distribution, i.e. tick arrivals form
    a Poisson process.
    """

    def __init__(
        self,
        initial_mid_price: float = 100.0,
        tick_volatility: float = 0.02,
        levels: int = DEFAULT_LEVELS,
        level_spacing: float = DEFAULT_LEVEL_SPACING,
        level_size: float = DEFAULT_LEVEL_SIZE,
        seed: int | None = None,
    ) -> None:
        self.mid_price = initial_mid_price
        self.tick_volatility = tick_volatility
        self.levels = levels
        self.level_spacing = level_spacing
        self.level_size = level_size
        self.order_book = OrderBook()
        self._rng = np.random.default_rng(seed)

    def step(self, timestamp: datetime) -> SyntheticTick:
        """Advance the midpoint by one Brownian increment and re-quote resting
        liquidity on both sides of the book."""
        self.mid_price = max(0.01, self.mid_price + self._rng.normal(0, self.tick_volatility))
        bounce = self._rng.normal(0, self.level_spacing / 2)

        self._cancel_all_resting()
        self._add_levels("BUY", timestamp, bounce)
        self._add_levels("SELL", timestamp, bounce)

        return SyntheticTick(
            timestamp=timestamp,
            mid_price=self.mid_price,
            best_bid=self.order_book.best_bid(),
            best_ask=self.order_book.best_ask(),
        )

    def _cancel_all_resting(self) -> None:
        for book in (self.order_book.bids, self.order_book.asks):
            order_ids = [order.order_id for level in list(book.values()) for order in list(level)]
            for order_id in order_ids:
                self.order_book.cancel(order_id)

    def _add_levels(self, side: str, timestamp: datetime, bounce: float) -> None:
        direction = -1 if side == "BUY" else 1
        for level_index in range(1, self.levels + 1):
            offset = direction * level_index * self.level_spacing + bounce
            price = round(self.mid_price + offset, 2)
            if price <= 0:
                continue
            self.order_book.add_limit(side, price, self.level_size, timestamp)

    def generate_ohlcv(
        self,
        n_ticks: int,
        start: datetime,
        mean_interval_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """Generate `n_ticks` tick-level OHLCV rows, in the same schema
        `HistoricDataHandler` expects, so it can replay tick data exactly like
        daily bars with no changes to the engine."""
        timestamps: list[datetime] = []
        rows: list[dict[str, float]] = []
        timestamp = start

        for _ in range(n_ticks):
            timestamp = timestamp + timedelta(seconds=float(self._rng.exponential(mean_interval_seconds)))
            tick = self.step(timestamp)
            timestamps.append(timestamp)
            rows.append(
                {
                    "open": tick.mid_price,
                    "high": tick.best_ask if tick.best_ask is not None else tick.mid_price,
                    "low": tick.best_bid if tick.best_bid is not None else tick.mid_price,
                    "close": tick.mid_price,
                    "volume": self.level_size,
                }
            )

        return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
