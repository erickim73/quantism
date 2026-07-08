from __future__ import annotations

from quantsim.engine.event_queue import (
    Event,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from quantsim.engine.execution import ExecutionHandler
from quantsim.engine.metrics import performance_summary
from quantsim.engine.portfolio import Portfolio
from quantsim.engine.strategy import Strategy

DEFAULT_ORDER_QUANTITY = 100


class Backtester:
    """Orchestrates the event-driven backtest loop.

    Wiring: the data handler pushes MarketEvents -> Strategy.on_data turns them
    into SignalEvents -> signals become fixed-size OrderEvents -> the
    ExecutionHandler turns orders into FillEvents -> Portfolio applies fills.
    Everything after signal generation flows through the ExecutionHandler
    abstraction, so swapping SimpleExecutionHandler for the order-book-based
    handler (Phase 2) requires no changes to this class, Strategy, or Portfolio.
    """

    def __init__(
        self,
        data_handler,
        strategy: Strategy,
        execution_handler: ExecutionHandler,
        portfolio: Portfolio,
        order_quantity: float = DEFAULT_ORDER_QUANTITY,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.order_quantity = order_quantity
        self.event_queue: EventQueue = data_handler.event_queue

        self.fills: list[FillEvent] = []
        self.trade_pnls: list[float] = []
        self.trade_notionals: list[float] = []
        self._latest_prices: dict[str, float] = {}

    def run(self) -> dict[str, float]:
        while self.data_handler.continue_backtest or self.event_queue:
            if not self.event_queue:
                self.data_handler.update_bars()
                if not self.event_queue:
                    break

            self._dispatch(self.event_queue.pop())

        return performance_summary(self.portfolio.equity_curve, self.trade_pnls, self.trade_notionals)

    def _dispatch(self, event: Event) -> None:
        if isinstance(event, MarketEvent):
            self._handle_market_event(event)
        elif isinstance(event, SignalEvent):
            self._handle_signal_event(event)
        elif isinstance(event, OrderEvent):
            self._handle_order_event(event)
        elif isinstance(event, FillEvent):
            self._handle_fill_event(event)

    def _handle_market_event(self, event: MarketEvent) -> None:
        self._latest_prices[event.symbol] = event.close
        self.data_handler.mark_current(event)
        for signal in self.strategy.on_data(event, self.data_handler):
            self.event_queue.push(signal)
        self.portfolio.record_equity(event.timestamp, self._latest_prices)

    def _handle_signal_event(self, event: SignalEvent) -> None:
        if event.direction == "EXIT":
            position = self.portfolio.positions.get(event.symbol)
            if position is None or position.quantity == 0:
                return
            direction = "SELL" if position.quantity > 0 else "BUY"
            quantity = abs(position.quantity)
        else:
            direction = "BUY" if event.direction == "LONG" else "SELL"
            quantity = self.order_quantity * event.strength

        self.event_queue.push(
            OrderEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                order_type="MARKET",
                direction=direction,
                quantity=quantity,
            )
        )

    def _handle_order_event(self, event: OrderEvent) -> None:
        fill = self.execution_handler.execute_order(event)
        if fill is not None:
            self.event_queue.push(fill)

    def _handle_fill_event(self, event: FillEvent) -> None:
        realized_delta = self.portfolio.update_fill(event)
        self.strategy.on_fill(event)
        self.fills.append(event)
        self.trade_notionals.append(event.fill_price * event.quantity)
        if realized_delta != 0:
            self.trade_pnls.append(realized_delta)
