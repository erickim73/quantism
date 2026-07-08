from __future__ import annotations

import numpy as np

from quantsim.engine.event_queue import FillEvent, MarketEvent, SignalEvent
from quantsim.engine.strategy import Strategy


class BollingerZScoreStrategy(Strategy):
    """Mean-reversion strategy using a rolling z-score of price vs its
    Bollinger mean/std.

    Goes long when the z-score drops below -entry_z (oversold), short when it
    rises above +entry_z (overbought), and exits once the z-score reverts to
    within exit_z of zero. Position side is tracked from actual fills, not
    optimistically at signal time.
    """

    def __init__(
        self,
        symbols: list[str],
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError("exit_z must be less than entry_z")
        super().__init__(symbols)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._position_side: dict[str, str | None] = {symbol: None for symbol in symbols}

    def on_data(self, event: MarketEvent, data) -> list[SignalEvent]:
        history = data.get_latest_bars(event.symbol, self.window - 1)
        closes = list(history["close"]) + [event.close]
        if len(closes) < self.window:
            return []

        window_closes = np.array(closes[-self.window :])
        mean = window_closes.mean()
        std = window_closes.std(ddof=0)
        if std == 0:
            return []
        z_score = (event.close - mean) / std

        side = self._position_side[event.symbol]

        if side is None:
            if z_score <= -self.entry_z:
                return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="LONG")]
            if z_score >= self.entry_z:
                return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="SHORT")]
            return []

        if abs(z_score) <= self.exit_z:
            return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="EXIT")]
        return []

    def on_fill(self, fill: FillEvent) -> None:
        current = self._position_side.get(fill.symbol)
        if fill.direction == "BUY":
            self._position_side[fill.symbol] = None if current == "SHORT" else "LONG"
        else:
            self._position_side[fill.symbol] = None if current == "LONG" else "SHORT"
