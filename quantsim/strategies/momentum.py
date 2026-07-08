from __future__ import annotations

from quantsim.engine.event_queue import FillEvent, MarketEvent, SignalEvent
from quantsim.engine.strategy import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """Long-only moving-average crossover.

    Goes long when the short-window SMA crosses above the long-window SMA,
    exits when it crosses back below. Position state is updated from actual
    fills (`on_fill`), not optimistically at signal time, so a signal is never
    re-emitted while an order for the same transition is still in flight.
    """

    def __init__(self, symbols: list[str], short_window: int = 10, long_window: int = 30) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
        self._in_position: dict[str, bool] = {symbol: False for symbol in symbols}

    def on_data(self, event: MarketEvent, data) -> list[SignalEvent]:
        history = data.get_latest_bars(event.symbol, self.long_window - 1)
        closes = list(history["close"]) + [event.close]
        if len(closes) < self.long_window:
            return []

        short_ma = sum(closes[-self.short_window :]) / self.short_window
        long_ma = sum(closes[-self.long_window :]) / self.long_window
        in_position = self._in_position[event.symbol]

        if short_ma > long_ma and not in_position:
            return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="LONG")]
        if short_ma < long_ma and in_position:
            return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="EXIT")]
        return []

    def on_fill(self, fill: FillEvent) -> None:
        self._in_position[fill.symbol] = fill.direction == "BUY"
