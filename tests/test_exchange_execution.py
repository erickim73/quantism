from datetime import datetime

import pytest

from quantsim.engine.event_queue import OrderEvent
from quantsim.exchange.execution import OrderBookExecutionHandler
from quantsim.exchange.order_book import OrderBook

T0 = datetime(2024, 1, 1, 9, 30)


def make_order(direction="BUY", order_type="MARKET", quantity=5, limit_price=None):
    return OrderEvent(
        timestamp=T0,
        symbol="AAPL",
        order_type=order_type,
        direction=direction,
        quantity=quantity,
        limit_price=limit_price,
    )


def test_market_buy_within_top_level_has_zero_slippage():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=0.0)

    fill = handler.execute_order(make_order(direction="BUY", quantity=5))

    assert fill.fill_price == pytest.approx(100.0)
    assert fill.slippage == pytest.approx(0.0)


def test_market_buy_walking_multiple_levels_has_positive_slippage():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=101.0, quantity=5, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=0.0)

    fill = handler.execute_order(make_order(direction="BUY", quantity=8))

    expected_avg_price = (5 * 100.0 + 3 * 101.0) / 8
    assert fill.fill_price == pytest.approx(expected_avg_price)
    assert fill.slippage == pytest.approx((expected_avg_price - 100.0) * 8)


def test_market_sell_slippage_measured_against_best_bid():
    book = OrderBook()
    book.add_limit("BUY", price=99.0, quantity=5, timestamp=T0)
    book.add_limit("BUY", price=98.0, quantity=5, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=0.0)

    fill = handler.execute_order(make_order(direction="SELL", quantity=8))

    expected_avg_price = (5 * 99.0 + 3 * 98.0) / 8
    assert fill.fill_price == pytest.approx(expected_avg_price)
    # Selling below best bid is a cost, so slippage should be positive.
    assert fill.slippage == pytest.approx((99.0 - expected_avg_price) * 8)


def test_commission_is_bps_of_notional():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=10, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=5.0)

    fill = handler.execute_order(make_order(direction="BUY", quantity=10))

    assert fill.commission == pytest.approx(100.0 * 10 * (5.0 / 10_000))


def test_market_order_against_empty_book_returns_none():
    book = OrderBook()
    handler = OrderBookExecutionHandler(book)

    assert handler.execute_order(make_order(direction="BUY", quantity=5)) is None


def test_limit_order_type_executes_marketable_portion():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=0.0)

    fill = handler.execute_order(make_order(direction="BUY", order_type="LIMIT", quantity=5, limit_price=100.0))

    assert fill.fill_price == pytest.approx(100.0)


def test_limit_order_that_only_rests_returns_none():
    book = OrderBook()
    handler = OrderBookExecutionHandler(book)

    fill = handler.execute_order(make_order(direction="BUY", order_type="LIMIT", quantity=5, limit_price=90.0))

    assert fill is None
    assert book.best_bid() == 90.0  # order rested on the book


def test_limit_order_without_limit_price_raises_value_error():
    handler = OrderBookExecutionHandler(OrderBook())

    with pytest.raises(ValueError):
        handler.execute_order(make_order(order_type="LIMIT", limit_price=None))


def test_ioc_order_without_limit_price_raises_value_error():
    handler = OrderBookExecutionHandler(OrderBook())

    with pytest.raises(ValueError):
        handler.execute_order(make_order(order_type="IOC", limit_price=None))


def test_ioc_order_discards_unfilled_remainder():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=3, timestamp=T0)
    handler = OrderBookExecutionHandler(book, commission_bps=0.0)

    fill = handler.execute_order(make_order(direction="BUY", order_type="IOC", quantity=10, limit_price=100.0))

    assert fill.quantity == pytest.approx(3)
    assert book.best_bid() is None  # remainder discarded, not resting


def test_unsupported_order_type_raises_value_error():
    handler = OrderBookExecutionHandler(OrderBook())

    with pytest.raises(ValueError):
        handler.execute_order(make_order(order_type="STOP"))
