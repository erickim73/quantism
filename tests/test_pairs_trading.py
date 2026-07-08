from datetime import datetime

import numpy as np
import pytest

from quantsim.engine.event_queue import MarketEvent
from quantsim.strategies.pairs_trading import PairsTradingStrategy, is_cointegrated


def make_event(symbol, close, ts):
    return MarketEvent(timestamp=ts, symbol=symbol, open=close, high=close, low=close, close=close, volume=100)


def test_exit_z_must_be_less_than_entry_z():
    with pytest.raises(ValueError):
        PairsTradingStrategy("A", "B", entry_z=1.0, exit_z=1.0)


def test_is_cointegrated_true_for_a_genuinely_cointegrated_pair():
    rng = np.random.default_rng(0)
    n = 200
    common = np.cumsum(rng.normal(0, 1, n))
    series_a = common + rng.normal(0, 0.5, n)
    series_b = 0.5 * common + rng.normal(0, 0.5, n)

    assert is_cointegrated(series_a, series_b) is True


def test_is_cointegrated_false_for_independent_random_walks():
    rng = np.random.default_rng(1)
    n = 200
    series_a = np.cumsum(rng.normal(0, 1, n))
    series_b = np.cumsum(rng.normal(0, 1, n))

    assert is_cointegrated(series_a, series_b) is False


def test_on_data_returns_no_signal_with_insufficient_history():
    strategy = PairsTradingStrategy("A", "B", window=10)
    ts = datetime(2024, 1, 1)

    signals = strategy.on_data(make_event("A", 100.0, ts), data=None)

    assert signals == []


def test_on_data_returns_no_signal_when_spread_has_zero_variance():
    strategy = PairsTradingStrategy("A", "B", window=4, entry_z=1.0, exit_z=0.5)
    ts = datetime(2024, 1, 1)
    prices = [10.0, 11.0, 12.0, 13.0]

    for i, price in enumerate(prices):
        strategy.on_data(make_event("B", price, ts.replace(day=1 + i)), data=None)
    signals = []
    for i, price in enumerate(prices):
        signals = strategy.on_data(make_event("A", price, ts.replace(day=1 + i)), data=None)

    assert signals == []


def test_on_data_emits_short_a_long_b_when_a_is_rich_vs_regression_fit():
    strategy = PairsTradingStrategy("A", "B", window=10, entry_z=2.0, exit_z=0.5)
    ts = datetime(2024, 1, 1)
    b_prices = [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    a_prices = b_prices.copy()
    a_prices[-1] = b_prices[-1] + 100  # A spikes far above its regression fit

    for i in range(9):
        strategy.on_data(make_event("B", b_prices[i], ts.replace(day=1 + i)), data=None)
        strategy.on_data(make_event("A", a_prices[i], ts.replace(day=1 + i)), data=None)

    strategy.on_data(make_event("B", b_prices[-1], ts.replace(day=10)), data=None)
    signals = strategy.on_data(make_event("A", a_prices[-1], ts.replace(day=10)), data=None)

    assert len(signals) == 2
    by_symbol = {s.symbol: s for s in signals}
    assert by_symbol["A"].direction == "SHORT"
    assert by_symbol["B"].direction == "LONG"
    assert by_symbol["B"].strength == pytest.approx(6.4545454545, rel=1e-6)
    assert strategy._position_side == "SHORT_A_LONG_B"


def test_on_data_ignores_second_callback_on_the_same_day():
    strategy = PairsTradingStrategy("A", "B", window=10, entry_z=2.0, exit_z=0.5)
    ts = datetime(2024, 1, 1)
    b_prices = [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    a_prices = b_prices.copy()
    a_prices[-1] = b_prices[-1] + 100

    for i in range(9):
        strategy.on_data(make_event("B", b_prices[i], ts.replace(day=1 + i)), data=None)
        strategy.on_data(make_event("A", a_prices[i], ts.replace(day=1 + i)), data=None)

    last_day = ts.replace(day=10)
    first_call_signals = strategy.on_data(make_event("B", b_prices[-1], last_day), data=None)
    second_call_signals = strategy.on_data(make_event("A", a_prices[-1], last_day), data=None)

    # Whichever leg's callback happens to fire second on the same day should
    # not re-evaluate (and re-emit) the pair signal.
    assert first_call_signals == [] or second_call_signals == []


def test_on_data_emits_exit_signals_once_spread_reverts_near_zero():
    strategy = PairsTradingStrategy("A", "B", window=10, entry_z=2.0, exit_z=0.5)
    strategy._position_side = "LONG_A_SHORT_B"  # simulate an already-open position
    ts = datetime(2024, 1, 1)
    b_prices = [100.0, 101, 99, 102, 98, 103, 97, 104, 96, 100]
    noise = [2, -1, 3, -2, 1, -3, 2, -1, 0.5, 0.1]
    a_prices = [b + n for b, n in zip(b_prices, noise)]

    for i in range(9):
        strategy.on_data(make_event("B", b_prices[i], ts.replace(day=1 + i)), data=None)
        strategy.on_data(make_event("A", a_prices[i], ts.replace(day=1 + i)), data=None)

    strategy.on_data(make_event("B", b_prices[-1], ts.replace(day=10)), data=None)
    signals = strategy.on_data(make_event("A", a_prices[-1], ts.replace(day=10)), data=None)

    assert len(signals) == 2
    assert all(s.direction == "EXIT" for s in signals)
    assert {s.symbol for s in signals} == {"A", "B"}
    assert strategy._position_side is None
