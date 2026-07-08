from datetime import datetime

from quantsim.exchange.tick_generator import SyntheticTickGenerator

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
