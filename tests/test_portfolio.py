from datetime import datetime

import pytest

from quantsim.engine.event_queue import FillEvent
from quantsim.engine.portfolio import Portfolio


def make_fill(symbol, direction, quantity, fill_price, commission=0.0, ts=None):
    return FillEvent(
        timestamp=ts or datetime(2024, 1, 1),
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
    )


def test_buy_deducts_cash_and_opens_long_position():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])

    portfolio.update_fill(make_fill("AAPL", "BUY", 10, 100.0, commission=1.0))

    assert portfolio.cash == pytest.approx(10_000 - 1000 - 1.0)
    position = portfolio.positions["AAPL"]
    assert position.quantity == 10
    assert position.avg_price == pytest.approx(100.0)
    assert position.realized_pnl == 0.0


def test_full_close_realizes_pnl_and_flattens_position():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])
    portfolio.update_fill(make_fill("AAPL", "BUY", 10, 100.0))

    portfolio.update_fill(make_fill("AAPL", "SELL", 10, 110.0))

    position = portfolio.positions["AAPL"]
    assert position.quantity == 0
    assert position.avg_price == 0.0
    assert position.realized_pnl == pytest.approx(100.0)  # 10 * (110 - 100)


def test_partial_close_realizes_proportional_pnl_and_keeps_remainder():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])
    portfolio.update_fill(make_fill("AAPL", "BUY", 10, 100.0))

    portfolio.update_fill(make_fill("AAPL", "SELL", 4, 120.0))

    position = portfolio.positions["AAPL"]
    assert position.quantity == 6
    assert position.avg_price == pytest.approx(100.0)  # remainder keeps original cost basis
    assert position.realized_pnl == pytest.approx(4 * (120.0 - 100.0))


def test_position_flip_realizes_pnl_on_old_leg_and_opens_new_leg_at_fill_price():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])
    portfolio.update_fill(make_fill("AAPL", "BUY", 5, 100.0))

    portfolio.update_fill(make_fill("AAPL", "SELL", 8, 90.0))

    position = portfolio.positions["AAPL"]
    assert position.quantity == -3
    assert position.avg_price == pytest.approx(90.0)
    assert position.realized_pnl == pytest.approx(5 * (90.0 - 100.0))


def test_short_sell_from_flat_opens_short_position():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])

    portfolio.update_fill(make_fill("AAPL", "SELL", 5, 50.0))

    position = portfolio.positions["AAPL"]
    assert position.quantity == -5
    assert position.avg_price == pytest.approx(50.0)
    assert portfolio.cash == pytest.approx(10_000 + 250)


def test_unrealized_pnl_and_market_value_use_current_prices():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])
    portfolio.update_fill(make_fill("AAPL", "BUY", 10, 100.0))

    prices = {"AAPL": 130.0}

    assert portfolio.market_value(prices) == pytest.approx(1300.0)
    assert portfolio.unrealized_pnl(prices) == pytest.approx(300.0)
    assert portfolio.total_equity(prices) == pytest.approx(portfolio.cash + 1300.0)


def test_record_equity_appends_timestamped_equity_snapshot():
    portfolio = Portfolio(initial_cash=10_000, symbols=["AAPL"])
    ts = datetime(2024, 1, 2)

    portfolio.record_equity(ts, {"AAPL": 0.0})

    assert portfolio.equity_curve == [(ts, 10_000)]
