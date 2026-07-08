from __future__ import annotations

from quantsim.engine.event_queue import FillEvent, OrderEvent
from quantsim.engine.execution import ExecutionHandler
from quantsim.exchange.order_book import OrderBook


class OrderBookExecutionHandler(ExecutionHandler):
    """Fills orders by matching them against a live OrderBook, so fill price
    and slippage emerge from real order-book depth instead of a fixed bps
    model. Implements the same ExecutionHandler interface as
    engine.execution.SimpleExecutionHandler, so swapping between the two
    requires no changes to Strategy, Portfolio, or Backtester code.
    """

    def __init__(self, order_book: OrderBook, commission_bps: float = 1.0) -> None:
        self.order_book = order_book
        self.commission_bps = commission_bps

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        # Top-of-book price the order "should" have paid, for slippage attribution.
        reference_price = self.order_book.best_ask() if order.direction == "BUY" else self.order_book.best_bid()

        if order.order_type == "MARKET":
            trades, _ = self.order_book.market_order(order.direction, order.quantity, order.timestamp)
        elif order.order_type == "IOC":
            if order.limit_price is None:
                raise ValueError("IOC orders require a limit_price")
            trades, _ = self.order_book.ioc_order(
                order.direction, order.quantity, order.limit_price, order.timestamp
            )
        elif order.order_type == "LIMIT":
            if order.limit_price is None:
                raise ValueError("LIMIT orders require a limit_price")
            _, trades = self.order_book.add_limit(
                order.direction, order.limit_price, order.quantity, order.timestamp
            )
        else:
            raise ValueError(f"Unsupported order_type: {order.order_type!r}")

        if not trades:
            return None

        filled_qty = sum(trade.quantity for trade in trades)
        notional = sum(trade.price * trade.quantity for trade in trades)
        avg_price = notional / filled_qty
        commission = notional * (self.commission_bps / 10_000)

        slippage = 0.0
        if reference_price is not None:
            direction_sign = 1 if order.direction == "BUY" else -1
            slippage = direction_sign * (avg_price - reference_price) * filled_qty

        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=filled_qty,
            fill_price=avg_price,
            commission=commission,
            slippage=slippage,
        )
