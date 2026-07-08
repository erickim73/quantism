from datetime import datetime, timedelta

import pytest

from quantsim.exchange.order_book import OrderBook

T0 = datetime(2024, 1, 1, 9, 30)


def test_add_limit_rests_when_not_marketable():
    book = OrderBook()

    order_id, trades = book.add_limit("BUY", price=99.0, quantity=10, timestamp=T0)

    assert trades == []
    assert book.best_bid() == 99.0
    assert book.best_ask() is None


def test_best_bid_and_best_ask_reflect_top_of_book():
    book = OrderBook()
    book.add_limit("BUY", price=99.0, quantity=10, timestamp=T0)
    book.add_limit("BUY", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=102.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=101.0, quantity=5, timestamp=T0)

    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0


def test_marketable_limit_order_crosses_spread_immediately():
    book = OrderBook()
    book.add_limit("SELL", price=101.0, quantity=10, timestamp=T0)

    order_id, trades = book.add_limit("BUY", price=101.0, quantity=4, timestamp=T0 + timedelta(seconds=1))

    assert len(trades) == 1
    assert trades[0].price == 101.0
    assert trades[0].quantity == 4
    # No remainder should rest since the whole order was filled.
    assert book.best_bid() is None


def test_marketable_limit_order_rests_unfilled_remainder():
    book = OrderBook()
    book.add_limit("SELL", price=101.0, quantity=3, timestamp=T0)

    order_id, trades = book.add_limit("BUY", price=101.0, quantity=10, timestamp=T0 + timedelta(seconds=1))

    assert sum(t.quantity for t in trades) == 3
    assert book.best_bid() == 101.0  # remaining 7 shares now resting as a bid


def test_price_time_priority_fills_earlier_order_first_at_same_price():
    book = OrderBook()
    first_id, _ = book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    second_id, _ = book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0 + timedelta(seconds=1))

    trades, unfilled = book.market_order("BUY", quantity=5, timestamp=T0 + timedelta(seconds=2))

    assert unfilled == 0
    assert len(trades) == 1
    assert trades[0].resting_order_id == first_id  # FIFO: earlier order fills first


def test_market_order_walks_multiple_price_levels_for_large_size():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=101.0, quantity=5, timestamp=T0)

    trades, unfilled = book.market_order("BUY", quantity=8, timestamp=T0 + timedelta(seconds=1))

    assert unfilled == 0
    assert len(trades) == 2
    assert trades[0].price == 100.0
    assert trades[0].quantity == 5
    assert trades[1].price == 101.0
    assert trades[1].quantity == 3


def test_market_order_against_empty_book_returns_all_unfilled():
    book = OrderBook()

    trades, unfilled = book.market_order("BUY", quantity=10, timestamp=T0)

    assert trades == []
    assert unfilled == 10


def test_ioc_order_only_fills_at_price_or_better_and_discards_remainder():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=105.0, quantity=5, timestamp=T0)

    trades, unfilled = book.ioc_order("BUY", quantity=10, price=100.0, timestamp=T0 + timedelta(seconds=1))

    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert unfilled == 5
    # The IOC remainder must not rest on the book.
    assert book.best_bid() is None
    # The unmatched 105.0 ask should remain untouched.
    assert book.best_ask() == 105.0


def test_cancel_removes_resting_order_and_cleans_up_empty_level():
    book = OrderBook()
    order_id, _ = book.add_limit("BUY", price=99.0, quantity=10, timestamp=T0)

    result = book.cancel(order_id)

    assert result is True
    assert book.best_bid() is None


def test_cancel_of_unknown_order_id_returns_false():
    book = OrderBook()

    assert book.cancel(999) is False


def test_book_never_ends_up_crossed_after_marketable_orders():
    book = OrderBook()
    book.add_limit("SELL", price=100.0, quantity=10, timestamp=T0)
    book.add_limit("BUY", price=101.0, quantity=4, timestamp=T0 + timedelta(seconds=1))

    best_bid = book.best_bid()
    best_ask = book.best_ask()
    if best_bid is not None and best_ask is not None:
        assert best_bid < best_ask
