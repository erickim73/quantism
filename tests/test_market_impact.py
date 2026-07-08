from datetime import datetime

import pytest

from quantsim.engine.event_queue import FillEvent, OrderEvent
from quantsim.exchange.market_impact import (
    MarketImpactExecutionHandler,
    attribute_slippage,
    calibrate_y_coefficient,
    square_root_impact,
)
from quantsim.exchange.tick_generator import SyntheticTickGenerator

T0 = datetime(2024, 1, 1, 9, 30)


def test_square_root_impact_matches_formula_for_known_inputs():
    impact = square_root_impact(order_size=10_000, average_daily_volume=1_000_000, daily_volatility=0.02, y_coefficient=1.0)

    assert impact == pytest.approx(1.0 * 0.02 * (10_000 / 1_000_000) ** 0.5)


def test_square_root_impact_scales_with_square_root_of_size():
    base = square_root_impact(order_size=10_000, average_daily_volume=1_000_000, daily_volatility=0.02)
    quadrupled = square_root_impact(order_size=40_000, average_daily_volume=1_000_000, daily_volatility=0.02)

    assert quadrupled == pytest.approx(2 * base)


def test_square_root_impact_uses_absolute_order_size():
    positive = square_root_impact(order_size=5_000, average_daily_volume=1_000_000, daily_volatility=0.02)
    negative = square_root_impact(order_size=-5_000, average_daily_volume=1_000_000, daily_volatility=0.02)

    assert positive == pytest.approx(negative)


def test_square_root_impact_raises_for_non_positive_adv():
    with pytest.raises(ValueError):
        square_root_impact(order_size=1_000, average_daily_volume=0, daily_volatility=0.02)


def test_calibrate_y_coefficient_recovers_true_y_with_no_noise():
    true_y = 0.63
    adv = 2_000_000
    sigma = 0.015
    sizes = [1_000, 5_000, 20_000, 50_000]
    observed = [square_root_impact(s, adv, sigma, true_y) for s in sizes]

    calibrated = calibrate_y_coefficient(observed, sizes, adv, sigma)

    assert calibrated == pytest.approx(true_y, rel=1e-9)


def test_calibrate_y_coefficient_returns_zero_for_all_zero_sizes():
    calibrated = calibrate_y_coefficient([0.0, 0.0], [0, 0], average_daily_volume=1_000_000, daily_volatility=0.02)

    assert calibrated == 0.0


def test_attribute_slippage_components_sum_to_total_cost_vs_decision_price():
    attribution = attribute_slippage(
        fill_price=101.5,
        quantity=100,
        direction="BUY",
        mid_price_at_decision=100.0,
        mid_price_at_arrival=100.2,
        best_price_at_arrival=100.3,
    )

    expected_total = (101.5 - 100.0) * 100
    assert attribution.total == pytest.approx(expected_total)
    assert attribution.timing_cost == pytest.approx((100.2 - 100.0) * 100)
    assert attribution.spread_cost == pytest.approx((100.3 - 100.2) * 100)
    assert attribution.impact_cost == pytest.approx((101.5 - 100.3) * 100)


def test_attribute_slippage_sell_direction_flips_sign_convention():
    attribution = attribute_slippage(
        fill_price=98.5,
        quantity=100,
        direction="SELL",
        mid_price_at_decision=100.0,
        mid_price_at_arrival=99.8,
        best_price_at_arrival=99.7,
    )

    # For a sell, a lower fill price than the decision mid is a cost (positive).
    expected_total = (100.0 - 98.5) * 100
    assert attribution.total == pytest.approx(expected_total)


class _StubInnerHandler:
    def __init__(self, fill):
        self._fill = fill

    def execute_order(self, order):
        return self._fill


def test_market_impact_handler_shifts_mid_price_after_buy_fill():
    generator = SyntheticTickGenerator(initial_mid_price=100.0, seed=1)
    fill = FillEvent(timestamp=T0, symbol="AAPL", direction="BUY", quantity=50_000, fill_price=100.5, commission=0.0)
    handler = MarketImpactExecutionHandler(
        inner=_StubInnerHandler(fill),
        tick_generator=generator,
        average_daily_volume=1_000_000,
        daily_volatility=0.02,
        y_coefficient=1.0,
    )
    order = OrderEvent(timestamp=T0, symbol="AAPL", order_type="MARKET", direction="BUY", quantity=50_000)

    result = handler.execute_order(order)

    expected_impact = square_root_impact(50_000, 1_000_000, 0.02, 1.0)
    expected_mid = 100.0 + expected_impact * 100.0
    assert result is fill
    assert generator.mid_price == pytest.approx(expected_mid)
    assert handler.impact_log == [pytest.approx(expected_impact * 100.0)]


def test_market_impact_handler_shifts_mid_price_down_after_sell_fill():
    generator = SyntheticTickGenerator(initial_mid_price=100.0, seed=1)
    fill = FillEvent(timestamp=T0, symbol="AAPL", direction="SELL", quantity=50_000, fill_price=99.5, commission=0.0)
    handler = MarketImpactExecutionHandler(
        inner=_StubInnerHandler(fill),
        tick_generator=generator,
        average_daily_volume=1_000_000,
        daily_volatility=0.02,
        y_coefficient=1.0,
    )
    order = OrderEvent(timestamp=T0, symbol="AAPL", order_type="MARKET", direction="SELL", quantity=50_000)

    handler.execute_order(order)

    assert generator.mid_price < 100.0


def test_market_impact_handler_passes_through_none_without_shifting_price():
    generator = SyntheticTickGenerator(initial_mid_price=100.0, seed=1)
    handler = MarketImpactExecutionHandler(
        inner=_StubInnerHandler(None),
        tick_generator=generator,
        average_daily_volume=1_000_000,
        daily_volatility=0.02,
    )
    order = OrderEvent(timestamp=T0, symbol="AAPL", order_type="MARKET", direction="BUY", quantity=1_000)

    result = handler.execute_order(order)

    assert result is None
    assert generator.mid_price == pytest.approx(100.0)
    assert handler.impact_log == []
