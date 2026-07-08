from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from quantsim.engine.event_queue import FillEvent, MarketEvent, SignalEvent


class BarLookup(Protocol):
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame: ...


class Strategy(ABC):
    """Base class for trading strategies.

    `on_data` is called once per new `MarketEvent` for a symbol this strategy
    trades; it may inspect `data.get_latest_bars(symbol, n)` for bars prior to
    the current one (never later bars) and returns zero or more signals.
    `on_fill` lets the strategy react to its own order fills; it is a no-op
    unless overridden.
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    @abstractmethod
    def on_data(self, event: MarketEvent, data: BarLookup) -> list[SignalEvent]:
        """Handle a new bar for one symbol and return any resulting signals."""

    def on_fill(self, fill: FillEvent) -> None:
        return None
