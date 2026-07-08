from datetime import datetime

import pytest

from quantsim.exchange.tick_generator import SyntheticTickDataHandler, SyntheticTickGenerator

T0 = datetime(2024, 1, 1, 9, 30)


def test_seeded_generators_produce_identical_mid_price_paths():
    gen_a = SyntheticTickGenerator(seed=7)
    gen_b = SyntheticTickGenerator(seed=7)

    path_a = [gen_a.step(T0).mid_price for _ in range(20)]
    path_b = [gen_b.step(T0).mid_price for _ in range(20)]

    assert path_a == path_b


def test_step_never_leaves_book_crossed():
    generator = SyntheticTickGenerator(seed=1)

    for _ in range(50):
        tick = generator.step(T0)
        if tick.best_bid is not None and tick.best_ask is not None:
            assert tick.best_bid < tick.best_ask


def test_step_quotes_expected_number_of_levels_per_side():
    generator = SyntheticTickGenerator(seed=1, levels=5)

    generator.step(T0)

    assert len(generator.order_book.bids) == 5
    assert len(generator.order_book.asks) == 5


def test_generate_ohlcv_returns_expected_row_count_and_schema():
    generator = SyntheticTickGenerator(seed=2)

    frame = generator.generate_ohlcv(n_ticks=30, start=T0, mean_interval_seconds=0.5)

    assert len(frame) == 30
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_generate_ohlcv_timestamps_are_strictly_increasing_and_after_start():
    generator = SyntheticTickGenerator(seed=3)

    frame = generator.generate_ohlcv(n_ticks=15, start=T0, mean_interval_seconds=1.0)

    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert all(ts > T0 for ts in frame.index)


def test_generate_ohlcv_high_is_never_below_low():
    generator = SyntheticTickGenerator(seed=4)

    frame = generator.generate_ohlcv(n_ticks=25, start=T0, mean_interval_seconds=1.0)

    assert (frame["high"] >= frame["low"]).all()


def test_data_handler_update_bars_pushes_one_event_and_keeps_book_live():
    generator = SyntheticTickGenerator(seed=5)
    handler = SyntheticTickDataHandler(symbol="AAPL", generator=generator, n_ticks=3, start=T0)

    handler.update_bars()

    assert len(handler.event_queue) == 1
    # The generator's own book (not a stale pre-generated snapshot) reflects "now".
    assert generator.order_book.best_bid() is not None


def test_data_handler_sets_continue_backtest_false_after_last_tick():
    generator = SyntheticTickGenerator(seed=6)
    handler = SyntheticTickDataHandler(symbol="AAPL", generator=generator, n_ticks=2, start=T0)

    handler.update_bars()
    assert handler.continue_backtest is True
    handler.update_bars()
    assert handler.continue_backtest is False

    handler.update_bars()  # calling again past the end should not push more events
    assert len(handler.event_queue) == 2


def test_data_handler_get_latest_bars_excludes_marked_current_tick():
    generator = SyntheticTickGenerator(seed=8)
    handler = SyntheticTickDataHandler(symbol="AAPL", generator=generator, n_ticks=3, start=T0)

    handler.update_bars()
    first_event = handler.event_queue.pop()
    handler.mark_current(first_event)
    assert len(handler.get_latest_bars("AAPL", n=5)) == 0

    handler.update_bars()
    second_event = handler.event_queue.pop()
    handler.mark_current(second_event)
    latest = handler.get_latest_bars("AAPL", n=5)
    assert len(latest) == 1  # only the first tick is now "history"


def test_data_handler_get_next_bar_is_not_supported():
    generator = SyntheticTickGenerator(seed=9)
    handler = SyntheticTickDataHandler(symbol="AAPL", generator=generator, n_ticks=1, start=T0)

    with pytest.raises(NotImplementedError):
        handler.get_next_bar("AAPL", after=T0)


def test_next_arrival_time_is_strictly_after_current_and_deterministic_with_seed():
    gen_a = SyntheticTickGenerator(seed=10)
    gen_b = SyntheticTickGenerator(seed=10)

    next_a = gen_a.next_arrival_time(T0, mean_interval_seconds=2.0)
    next_b = gen_b.next_arrival_time(T0, mean_interval_seconds=2.0)

    assert next_a > T0
    assert next_a == next_b
