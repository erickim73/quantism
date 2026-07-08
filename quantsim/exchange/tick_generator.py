from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantsim.engine.event_queue import EventQueue, MarketEvent
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

    def next_arrival_time(self, current: datetime, mean_interval_seconds: float = 1.0) -> datetime:
        """Draw the next Poisson-process arrival time after `current`."""
        return current + timedelta(seconds=float(self._rng.exponential(mean_interval_seconds)))

    def generate_ohlcv(
        self,
        n_ticks: int,
        start: datetime,
        mean_interval_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """Generate `n_ticks` tick-level OHLCV rows, in the same schema
        `HistoricDataHandler` expects, so it can replay tick data exactly like
        daily bars with no changes to the engine.

        This pre-generates the *entire* stream, which advances `self.order_book`
        all the way to its final state — fine for a fixed-cost execution model
        (e.g. SimpleExecutionHandler) that only reads OHLCV values, but wrong
        for live order-book execution against `self.order_book` mid-replay.
        For that, use `SyntheticTickDataHandler`, which steps the generator
        incrementally so the book stays in sync with "now".
        """
        timestamps: list[datetime] = []
        rows: list[dict[str, float]] = []
        timestamp = start

        for _ in range(n_ticks):
            timestamp = self.next_arrival_time(timestamp, mean_interval_seconds)
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


class SyntheticTickDataHandler:
    """Drives an EventQueue with a synthetic tick stream generated on demand —
    one tick per `update_bars()` call — so `generator.order_book` always
    reflects the correct state for "now" during replay.

    This is the live counterpart to `SyntheticTickGenerator.generate_ohlcv`,
    which pre-generates the whole stream up front (leaving the book stuck in
    its *final* state). Use this handler with
    `exchange.execution.OrderBookExecutionHandler`, which matches orders
    immediately against the book's current state; `get_next_bar` is not
    supported here since there is no "next bar" to look ahead to yet.
    """

    def __init__(
        self,
        symbol: str,
        generator: SyntheticTickGenerator,
        n_ticks: int,
        start: datetime,
        mean_interval_seconds: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.symbols = [symbol]
        self.generator = generator
        self.n_ticks = n_ticks
        self.mean_interval_seconds = mean_interval_seconds
        self.event_queue: EventQueue = EventQueue()
        self._timestamp = start
        self._ticks_generated = 0
        self._history: list[MarketEvent] = []
        self._current_time: datetime | None = None
        self.continue_backtest = True

    def update_bars(self) -> None:
        if self._ticks_generated >= self.n_ticks:
            self.continue_backtest = False
            return

        self._timestamp = self.generator.next_arrival_time(self._timestamp, self.mean_interval_seconds)
        tick = self.generator.step(self._timestamp)
        event = MarketEvent(
            timestamp=self._timestamp,
            symbol=self.symbol,
            open=tick.mid_price,
            high=tick.best_ask if tick.best_ask is not None else tick.mid_price,
            low=tick.best_bid if tick.best_bid is not None else tick.mid_price,
            close=tick.mid_price,
            volume=self.generator.level_size,
        )
        self._history.append(event)
        self._ticks_generated += 1
        self.event_queue.push(event)

        if self._ticks_generated >= self.n_ticks:
            self.continue_backtest = False

    def mark_current(self, event: MarketEvent) -> None:
        self._current_time = event.timestamp

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        if self._current_time is None:
            window: list[MarketEvent] = []
        else:
            history_before = [e for e in self._history if e.timestamp < self._current_time]
            window = history_before[-n:] if n > 0 else []
        return pd.DataFrame(
            [{"open": e.open, "high": e.high, "low": e.low, "close": e.close, "volume": e.volume} for e in window]
        )

    def get_next_bar(self, symbol: str, after: datetime) -> MarketEvent | None:
        raise NotImplementedError(
            "SyntheticTickDataHandler executes immediately against the live order "
            "book (see OrderBookExecutionHandler); it does not support next-bar "
            "lookahead fills."
        )
