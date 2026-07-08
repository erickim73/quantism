from datetime import datetime

import pandas as pd
import pytest

from quantsim.engine.event_queue import FillEvent, MarketEvent
from quantsim.strategies.mean_reversion import BollingerZScoreStrategy


class _StubDataHandler:
    def __init__(self, closes):
        self._closes = closes

    def get_latest_bars(self, symbol, n=1):
        window = self._closes[-n:] if n > 0 else []
        return pd.DataFrame({"close": window})


def make_event(close, symbol="AAPL"):
    return MarketEvent(
        timestamp=datetime(2024, 1, 1), symbol=symbol, open=close, high=close, low=close, close=close, volume=100
    )


def test_exit_z_must_be_less_than_entry_z():
    with pytest.raises(ValueError):
        BollingerZScoreStrategy(symbols=["AAPL"], entry_z=1.0, exit_z=1.0)


def test_returns_no_signal_with_insufficient_history():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    data = _StubDataHandler(closes=[10.0, 10.0])  # only 2 prior, need 3

    assert strategy.on_data(make_event(10.0), data) == []


def test_returns_no_signal_when_window_has_zero_variance():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    data = _StubDataHandler(closes=[10.0, 10.0, 10.0])

    assert strategy.on_data(make_event(10.0), data) == []


def test_emits_long_signal_when_oversold():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    data = _StubDataHandler(closes=[10.0, 10.0, 10.0])

    signals = strategy.on_data(make_event(5.0), data)  # z ~= -1.73

    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_emits_short_signal_when_overbought():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    data = _StubDataHandler(closes=[10.0, 10.0, 10.0])

    signals = strategy.on_data(make_event(15.0), data)  # z ~= +1.73

    assert len(signals) == 1
    assert signals[0].direction == "SHORT"


def test_no_repeated_signal_while_still_in_extreme_zone_and_already_positioned():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=5, commission=0))
    data = _StubDataHandler(closes=[10.0, 10.0, 10.0])

    assert strategy.on_data(make_event(5.0), data) == []


def test_emits_exit_signal_once_zscore_reverts_to_near_zero():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"], window=4, entry_z=1.0, exit_z=0.5)
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=5, commission=0))
    data = _StubDataHandler(closes=[10.0, 12.0, 8.0])

    signals = strategy.on_data(make_event(10.0), data)  # z == 0

    assert len(signals) == 1
    assert signals[0].direction == "EXIT"


def test_on_fill_buy_from_flat_opens_long():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"])
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=5, commission=0))
    assert strategy._position_side["AAPL"] == "LONG"


def test_on_fill_sell_from_flat_opens_short():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"])
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="SELL", quantity=10, fill_price=5, commission=0))
    assert strategy._position_side["AAPL"] == "SHORT"


def test_on_fill_sell_while_long_flattens_position():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"])
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="BUY", quantity=10, fill_price=5, commission=0))
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 2), symbol="AAPL", direction="SELL", quantity=10, fill_price=6, commission=0))
    assert strategy._position_side["AAPL"] is None


def test_on_fill_buy_while_short_flattens_position():
    strategy = BollingerZScoreStrategy(symbols=["AAPL"])
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 1), symbol="AAPL", direction="SELL", quantity=10, fill_price=5, commission=0))
    strategy.on_fill(FillEvent(timestamp=datetime(2024, 1, 2), symbol="AAPL", direction="BUY", quantity=10, fill_price=4, commission=0))
    assert strategy._position_side["AAPL"] is None
