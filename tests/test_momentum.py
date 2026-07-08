from datetime import datetime, timedelta

import pandas as pd
import pytest

from quantsim.engine.event_queue import FillEvent, MarketEvent
from quantsim.strategies.momentum import MovingAverageCrossoverStrategy


class _StubDataHandler:
    """Returns a fixed rolling window of closes regardless of symbol/n."""

    def __init__(self, closes):
        self._closes = closes

    def get_latest_bars(self, symbol, n=1):
        window = self._closes[-n:] if n > 0 else []
        return pd.DataFrame({"close": window})


def make_event(close, ts=None, symbol="AAPL"):
    ts = ts or datetime(2024, 1, 1)
    return MarketEvent(timestamp=ts, symbol=symbol, open=close, high=close, low=close, close=close, volume=100)


def test_short_window_must_be_less_than_long_window():
    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=10, long_window=5)


def test_returns_no_signal_with_insufficient_history():
    strategy = MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=4)
    data = _StubDataHandler(closes=[100.0, 101.0])  # only 2 prior bars, need 3

    signals = strategy.on_data(make_event(102.0), data)

    assert signals == []


def test_emits_long_signal_when_short_ma_crosses_above_long_ma():
    strategy = MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=4)
    # Prior 3 closes flat/declining, current bar spikes up so short MA > long MA.
    data = _StubDataHandler(closes=[100.0, 99.0, 98.0])

    signals = strategy.on_data(make_event(120.0), data)

    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_does_not_re_emit_long_signal_while_already_in_position():
    strategy = MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=4)
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=100, commission=0))
    data = _StubDataHandler(closes=[100.0, 99.0, 98.0])

    signals = strategy.on_data(make_event(120.0), data)

    assert signals == []


def test_emits_exit_signal_when_short_ma_crosses_below_long_ma_while_in_position():
    strategy = MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=4)
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=100, commission=0))
    data = _StubDataHandler(closes=[120.0, 115.0, 110.0])

    signals = strategy.on_data(make_event(50.0), data)

    assert len(signals) == 1
    assert signals[0].direction == "EXIT"


def test_on_fill_tracks_in_position_state():
    strategy = MovingAverageCrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=4)

    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=100, commission=0))
    assert strategy._in_position["AAPL"] is True

    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 2), symbol="AAPL", direction="SELL", quantity=10, fill_price=110, commission=0))
    assert strategy._in_position["AAPL"] is False
