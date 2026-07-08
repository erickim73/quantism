from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

from quantsim.engine.event_queue import FillEvent, MarketEvent, OrderEvent


class NextBarLookup(Protocol):
    def get_next_bar(self, symbol: str, after: datetime) -> MarketEvent | None: ...


class ExecutionHandler(ABC):
    """Turns an Order into a Fill.

    Strategy and Portfolio never depend on which concrete implementation is
    active: this naive fill-at-next-open handler, or the order-book-based
    handler in exchange/execution.py.
    """

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        """Return the resulting Fill, or None if the order cannot be filled yet."""


class SimpleExecutionHandler(ExecutionHandler):
    """Fills market orders at the next bar's open price with flat commission
    and slippage (both in basis points of notional). Limit/IOC orders are not
    modeled here; see exchange/execution.py for order-book-based fills.
    """

    def __init__(
        self,
        data_source: NextBarLookup,
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
    ) -> None:
        self.data_source = data_source
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        if order.order_type != "MARKET":
            raise NotImplementedError(
                f"SimpleExecutionHandler only supports MARKET orders, got {order.order_type!r}"
            )

        next_bar = self.data_source.get_next_bar(order.symbol, after=order.timestamp)
        if next_bar is None:
            return None

        direction_sign = 1 if order.direction == "BUY" else -1
        slippage_per_share = next_bar.open * (self.slippage_bps / 10_000)
        fill_price = next_bar.open + direction_sign * slippage_per_share
        commission = fill_price * order.quantity * (self.commission_bps / 10_000)

        return FillEvent(
            timestamp=next_bar.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_per_share * order.quantity,
        )
