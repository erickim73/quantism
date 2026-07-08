"""Cointegration-based pairs trading (statistical arbitrage).

Trades the spread between two symbols: a hedge ratio is estimated via OLS
regression of one price series on the other, and a rolling z-score of the
resulting spread drives entries/exits. `is_cointegrated` runs the
Engle-Granger test as a sanity check that a candidate pair actually has a
stable long-run equilibrium relationship before trading it.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import coint

from quantsim.engine.event_queue import MarketEvent, SignalEvent
from quantsim.engine.strategy import Strategy


def is_cointegrated(series_a: np.ndarray, series_b: np.ndarray, significance: float = 0.05) -> bool:
    """Engle-Granger cointegration test: is there a stable long-run
    equilibrium relationship between the two price series?"""
    _, p_value, _ = coint(series_a, series_b)
    return bool(p_value < significance)


class PairsTradingStrategy(Strategy):
    """Long-short mean-reversion on the spread between `symbol_a` and
    `symbol_b`.

    Assumes both symbols receive a bar on every replayed date (true for data
    produced by `data.loaders.align_frames`, which reindexes all symbols onto
    the union of trading dates), so their price histories stay index-aligned.
    Because `on_data` is called once per symbol per bar, a `_last_signal_date`
    guard prevents evaluating — and double-emitting — the same day's signal
    twice (once per leg's callback).

    The B leg is sized by `strength=hedge_ratio` (see
    `Backtester._handle_signal_event`) so the two legs are approximately
    dollar/beta-neutral rather than equal-share-count, which is how a real
    pairs trade is actually sized.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError("exit_z must be less than entry_z")
        super().__init__(symbols=[symbol_a, symbol_b])
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

        self._prices: dict[str, list[float]] = {symbol_a: [], symbol_b: []}
        self._position_side: str | None = None
        self._last_signal_date = None

    def on_data(self, event: MarketEvent, data) -> list[SignalEvent]:
        self._prices[event.symbol].append(event.close)

        price_a_history = self._prices[self.symbol_a]
        price_b_history = self._prices[self.symbol_b]
        n = min(len(price_a_history), len(price_b_history))
        if n < self.window or event.timestamp == self._last_signal_date:
            return []
        self._last_signal_date = event.timestamp

        window_a = np.array(price_a_history[-self.window :])
        window_b = np.array(price_b_history[-self.window :])
        hedge_ratio = float(np.polyfit(window_b, window_a, 1)[0])

        spread = window_a - hedge_ratio * window_b
        mean, std = spread.mean(), spread.std(ddof=0)
        if std == 0:
            return []
        z_score = (spread[-1] - mean) / std

        if self._position_side is None:
            if z_score >= self.entry_z:
                self._position_side = "SHORT_A_LONG_B"
                return [
                    SignalEvent(timestamp=event.timestamp, symbol=self.symbol_a, direction="SHORT"),
                    SignalEvent(
                        timestamp=event.timestamp, symbol=self.symbol_b, direction="LONG", strength=hedge_ratio
                    ),
                ]
            if z_score <= -self.entry_z:
                self._position_side = "LONG_A_SHORT_B"
                return [
                    SignalEvent(timestamp=event.timestamp, symbol=self.symbol_a, direction="LONG"),
                    SignalEvent(
                        timestamp=event.timestamp, symbol=self.symbol_b, direction="SHORT", strength=hedge_ratio
                    ),
                ]
            return []

        if abs(z_score) <= self.exit_z:
            self._position_side = None
            return [
                SignalEvent(timestamp=event.timestamp, symbol=self.symbol_a, direction="EXIT"),
                SignalEvent(timestamp=event.timestamp, symbol=self.symbol_b, direction="EXIT"),
            ]
        return []
