from datetime import datetime

import pytest

from quantsim.engine.event_queue import MarketEvent, OrderEvent
from quantsim.engine.execution import SimpleExecutionHandler


class _StubDataSource:
    def __init__(self, next_bar):
        self._next_bar = next_bar

    def get_next_bar(self, symbol, after):
        return self._next_bar


def make_order(direction="BUY", quantity=10, order_type="MARKET", ts=None):
    return OrderEvent(
        timestamp=ts or datetime(2024, 1, 1),
        symbol="AAPL",
        order_type=order_type,
        direction=direction,
        quantity=quantity,
    )


def make_next_bar(open_price=100.0, ts=None):
    return MarketEvent(
        timestamp=ts or datetime(2024, 1, 2),
        symbol="AAPL",
        open=open_price,
        high=open_price,
        low=open_price,
        close=open_price,
        volume=1000,
    )


def test_buy_order_fills_above_next_open_by_slippage():
    next_bar = make_next_bar(open_price=100.0)
    handler = SimpleExecutionHandler(_StubDataSource(next_bar), commission_bps=0.0, slippage_bps=10.0)

    fill = handler.execute_order(make_order(direction="BUY"))

    expected_slippage_per_share = 100.0 * (10.0 / 10_000)
    assert fill.fill_price == pytest.approx(100.0 + expected_slippage_per_share)
    assert fill.timestamp == next_bar.timestamp


def test_sell_order_fills_below_next_open_by_slippage():
    next_bar = make_next_bar(open_price=100.0)
    handler = SimpleExecutionHandler(_StubDataSource(next_bar), commission_bps=0.0, slippage_bps=10.0)

    fill = handler.execute_order(make_order(direction="SELL"))

    expected_slippage_per_share = 100.0 * (10.0 / 10_000)
    assert fill.fill_price == pytest.approx(100.0 - expected_slippage_per_share)


def test_commission_is_bps_of_fill_notional():
    next_bar = make_next_bar(open_price=100.0)
    handler = SimpleExecutionHandler(_StubDataSource(next_bar), commission_bps=5.0, slippage_bps=0.0)

    fill = handler.execute_order(make_order(direction="BUY", quantity=20))

    assert fill.fill_price == pytest.approx(100.0)
    assert fill.commission == pytest.approx(100.0 * 20 * (5.0 / 10_000))


def test_returns_none_when_no_next_bar_available():
    handler = SimpleExecutionHandler(_StubDataSource(None))

    assert handler.execute_order(make_order()) is None


def test_raises_not_implemented_for_non_market_orders():
    handler = SimpleExecutionHandler(_StubDataSource(make_next_bar()))

    with pytest.raises(NotImplementedError):
        handler.execute_order(make_order(order_type="LIMIT"))
