import pandas as pd
import pytest

from quantsim.data.loaders import HistoricDataHandler
from quantsim.engine.backtester import Backtester
from quantsim.engine.event_queue import EventQueue, SignalEvent
from quantsim.engine.execution import SimpleExecutionHandler
from quantsim.engine.portfolio import Portfolio
from quantsim.engine.strategy import Strategy


def make_frame(dates, opens):
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens,
            "volume": [1000] * len(opens),
        },
        index=index,
    )


class _ScriptedStrategy(Strategy):
    """Emits a pre-scripted signal on specific bar timestamps; used only for testing."""

    def __init__(self, symbols, signal_schedule):
        super().__init__(symbols)
        self.signal_schedule = signal_schedule
        self.observed_timestamps = []

    def on_data(self, event, data):
        self.observed_timestamps.append(event.timestamp)
        direction = self.signal_schedule.get(event.timestamp)
        if direction is None:
            return []
        return [SignalEvent(timestamp=event.timestamp, symbol=event.symbol, direction=direction)]


def test_full_backtest_buy_then_exit_realizes_expected_pnl():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    opens = [100.0, 101.0, 102.0, 103.0, 104.0]
    frames = {"TEST": make_frame(dates, opens)}

    queue = EventQueue()
    data_handler = HistoricDataHandler(frames, queue)
    strategy = _ScriptedStrategy(
        symbols=["TEST"],
        signal_schedule={
            pd.Timestamp("2024-01-01"): "LONG",
            pd.Timestamp("2024-01-03"): "EXIT",
        },
    )
    execution_handler = SimpleExecutionHandler(data_handler, commission_bps=0.0, slippage_bps=0.0)
    portfolio = Portfolio(initial_cash=10_000.0, symbols=["TEST"])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=10)
    summary = backtester.run()

    # LONG signal on 01-01 fills at next open (01-02, price 101); EXIT signal on
    # 01-03 fills at next open (01-04, price 103) -> realized pnl = 10 * (103-101).
    assert len(backtester.fills) == 2
    assert backtester.fills[0].fill_price == pytest.approx(101.0)
    assert backtester.fills[1].fill_price == pytest.approx(103.0)
    assert portfolio.positions["TEST"].quantity == 0
    assert portfolio.realized_pnl() == pytest.approx(20.0)
    assert portfolio.cash == pytest.approx(10_000.0 + 20.0)
    assert strategy.observed_timestamps == [pd.Timestamp(d) for d in dates]
    assert set(summary.keys()) == {"sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "turnover"}
    assert data_handler.continue_backtest is False


def test_signal_strength_scales_order_quantity():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    opens = [100.0, 101.0, 102.0]
    frames = {"TEST": make_frame(dates, opens)}

    queue = EventQueue()
    data_handler = HistoricDataHandler(frames, queue)
    strategy = _ScriptedStrategy(symbols=["TEST"], signal_schedule={})
    # Override on_data to emit a half-strength LONG signal on the first bar,
    # instead of using the schedule dict (which always uses strength=1.0).
    strategy.on_data = lambda event, data, _orig=strategy.on_data: (
        [SignalEvent(timestamp=event.timestamp, symbol="TEST", direction="LONG", strength=0.5)]
        if event.timestamp == pd.Timestamp("2024-01-01")
        else _orig(event, data)
    )
    execution_handler = SimpleExecutionHandler(data_handler, commission_bps=0.0, slippage_bps=0.0)
    portfolio = Portfolio(initial_cash=10_000.0, symbols=["TEST"])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio, order_quantity=20)
    backtester.run()

    assert len(backtester.fills) == 1
    assert backtester.fills[0].quantity == pytest.approx(10)  # 20 * 0.5


def test_backtest_with_no_signals_produces_flat_equity_curve():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    opens = [50.0, 50.0, 50.0]
    frames = {"TEST": make_frame(dates, opens)}

    queue = EventQueue()
    data_handler = HistoricDataHandler(frames, queue)
    strategy = _ScriptedStrategy(symbols=["TEST"], signal_schedule={})
    execution_handler = SimpleExecutionHandler(data_handler)
    portfolio = Portfolio(initial_cash=5_000.0, symbols=["TEST"])

    backtester = Backtester(data_handler, strategy, execution_handler, portfolio)
    backtester.run()

    assert len(backtester.fills) == 0
    assert portfolio.cash == pytest.approx(5_000.0)
    assert len(portfolio.equity_curve) == 3
