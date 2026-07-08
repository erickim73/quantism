from datetime import datetime, timedelta

import pytest

from quantsim.engine.event_queue import EventQueue, MarketEvent, SignalEvent


def make_market_event(ts: datetime, symbol: str = "AAPL") -> MarketEvent:
    return MarketEvent(
        timestamp=ts, symbol=symbol, open=1.0, high=1.0, low=1.0, close=1.0, volume=100
    )


def test_pop_returns_events_in_timestamp_order():
    queue = EventQueue()
    base = datetime(2024, 1, 1)
    e_later = make_market_event(base + timedelta(days=2))
    e_earlier = make_market_event(base)
    e_middle = make_market_event(base + timedelta(days=1))

    queue.push(e_later)
    queue.push(e_earlier)
    queue.push(e_middle)

    assert queue.pop() is e_earlier
    assert queue.pop() is e_middle
    assert queue.pop() is e_later


def test_same_timestamp_events_pop_in_insertion_order():
    queue = EventQueue()
    ts = datetime(2024, 1, 1)
    first = make_market_event(ts, symbol="AAPL")
    second = make_market_event(ts, symbol="MSFT")
    third = SignalEvent(timestamp=ts, symbol="AAPL", direction="LONG")

    queue.push(first)
    queue.push(second)
    queue.push(third)

    assert queue.pop() is first
    assert queue.pop() is second
    assert queue.pop() is third


def test_peek_does_not_remove_event():
    queue = EventQueue()
    event = make_market_event(datetime(2024, 1, 1))
    queue.push(event)

    assert queue.peek() is event
    assert len(queue) == 1
    assert queue.pop() is event


def test_len_and_bool_reflect_queue_size():
    queue = EventQueue()
    assert len(queue) == 0
    assert bool(queue) is False

    queue.push(make_market_event(datetime(2024, 1, 1)))
    assert len(queue) == 1
    assert bool(queue) is True


def test_pop_from_empty_queue_raises_index_error():
    queue = EventQueue()
    with pytest.raises(IndexError):
        queue.pop()


def test_peek_from_empty_queue_raises_index_error():
    queue = EventQueue()
    with pytest.raises(IndexError):
        queue.peek()
