"""Behavioral-parity tests between the pure-Python OrderBook and the optional
C++-accelerated CppOrderBook. Both implementations are run through the same
test bodies (parametrized over the class) to prove they're interchangeable,
not just individually correct.

Skipped entirely if the optional C++ extension isn't installed (`pip install
./cpp`) -- the main test suite must stay green without a C++ toolchain.
"""

from datetime import datetime, timedelta

import pytest

pytest.importorskip("quantsim_matching_engine")

from quantsim.exchange.cpp_order_book import CppOrderBook  # noqa: E402
from quantsim.exchange.order_book import OrderBook  # noqa: E402

T0 = datetime(2024, 1, 1, 9, 30)

BOOK_IMPLEMENTATIONS = [OrderBook, CppOrderBook]


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_add_limit_rests_when_not_marketable(book_cls):
    book = book_cls()

    order_id, trades = book.add_limit("BUY", price=99.0, quantity=10, timestamp=T0)

    assert trades == []
    assert book.best_bid() == 99.0
    assert book.best_ask() is None


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_marketable_limit_order_rests_unfilled_remainder(book_cls):
    book = book_cls()
    book.add_limit("SELL", price=101.0, quantity=3, timestamp=T0)

    order_id, trades = book.add_limit("BUY", price=101.0, quantity=10, timestamp=T0 + timedelta(seconds=1))

    assert sum(t.quantity for t in trades) == 3
    assert book.best_bid() == 101.0


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_price_time_priority_fills_earlier_order_first_at_same_price(book_cls):
    book = book_cls()
    first_id, _ = book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0 + timedelta(seconds=1))

    trades, unfilled = book.market_order("BUY", quantity=5, timestamp=T0 + timedelta(seconds=2))

    assert unfilled == 0
    assert len(trades) == 1
    assert trades[0].resting_order_id == first_id


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_market_order_walks_multiple_price_levels_for_large_size(book_cls):
    book = book_cls()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=101.0, quantity=5, timestamp=T0)

    trades, unfilled = book.market_order("BUY", quantity=8, timestamp=T0 + timedelta(seconds=1))

    assert unfilled == 0
    assert len(trades) == 2
    assert trades[0].price == 100.0
    assert trades[0].quantity == 5
    assert trades[1].price == 101.0
    assert trades[1].quantity == 3


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_ioc_order_only_fills_at_price_or_better_and_discards_remainder(book_cls):
    book = book_cls()
    book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    book.add_limit("SELL", price=105.0, quantity=5, timestamp=T0)

    trades, unfilled = book.ioc_order("BUY", quantity=10, price=100.0, timestamp=T0 + timedelta(seconds=1))

    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert unfilled == 5
    assert book.best_bid() is None
    assert book.best_ask() == 105.0


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_cancel_removes_resting_order_and_cleans_up_empty_level(book_cls):
    book = book_cls()
    order_id, _ = book.add_limit("BUY", price=99.0, quantity=10, timestamp=T0)

    assert book.cancel(order_id) is True
    assert book.best_bid() is None
    assert book.cancel(999) is False


@pytest.mark.parametrize("book_cls", BOOK_IMPLEMENTATIONS)
def test_book_never_ends_up_crossed_after_marketable_orders(book_cls):
    book = book_cls()
    book.add_limit("SELL", price=100.0, quantity=10, timestamp=T0)
    book.add_limit("BUY", price=101.0, quantity=4, timestamp=T0 + timedelta(seconds=1))

    best_bid = book.best_bid()
    best_ask = book.best_ask()
    if best_bid is not None and best_ask is not None:
        assert best_bid < best_ask


def test_run_batch_matches_equivalent_sequential_calls():
    sequential_book = CppOrderBook()
    sequential_book.add_limit("SELL", price=100.0, quantity=5, timestamp=T0)
    sequential_book.add_limit("SELL", price=101.0, quantity=5, timestamp=T0)
    trades, unfilled = sequential_book.market_order("BUY", quantity=8, timestamp=T0)

    batch_book = CppOrderBook()
    total_trades = batch_book.run_batch(
        [
            (False, "SELL", 100.0, 5.0),
            (False, "SELL", 101.0, 5.0),
            (True, "BUY", 0.0, 8.0),
        ]
    )

    assert total_trades == len(trades)
    assert sequential_book.best_bid() == batch_book.best_bid()
    assert sequential_book.best_ask() == batch_book.best_ask()
    assert unfilled == 0


def test_run_batch_rests_unfilled_limit_orders():
    book = CppOrderBook()

    total_trades = book.run_batch([(False, "BUY", 99.0, 10.0)])

    assert total_trades == 0
    assert book.best_bid() == 99.0
