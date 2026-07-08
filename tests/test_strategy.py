from datetime import datetime

import pandas as pd
import pytest

from quantsim.engine.event_queue import FillEvent, MarketEvent, SignalEvent
from quantsim.engine.strategy import Strategy


class _EchoStrategy(Strategy):
    """Minimal concrete strategy used only to exercise the base class."""

    def on_data(self, event: MarketEvent, data) -> list[SignalEvent]:
        return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction="LONG")]


class _StubDataHandler:
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        return pd.DataFrame()


def test_strategy_is_abstract_without_on_data():
    with pytest.raises(TypeError):
        Strategy(symbols=["AAPL"])  # type: ignore[abstract]


def test_concrete_strategy_returns_signals_from_on_data():
    strategy = _EchoStrategy(symbols=["AAPL"])
    event = MarketEvent(
        timestamp=datetime(2024, 1, 1),
        symbol="AAPL",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=100,
    )

    signals = strategy.on_data(event, _StubDataHandler())

    assert len(signals) == 1
    assert signals[0].direction == "LONG"
    assert signals[0].symbol == "AAPL"


def test_default_on_fill_is_a_no_op():
    strategy = _EchoStrategy(symbols=["AAPL"])
    fill = FillEvent(
        timestamp=datetime(2024, 1, 1),
        symbol="AAPL",
        direction="BUY",
        quantity=10,
        fill_price=100.0,
        commission=1.0,
    )

    assert strategy.on_fill(fill) is None
