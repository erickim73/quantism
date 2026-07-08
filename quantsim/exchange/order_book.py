from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sortedcontainers import SortedDict

Side = Literal["BUY", "SELL"]


@dataclass
class RestingOrder:
    order_id: int
    side: Side
    price: float
    quantity: float
    timestamp: datetime


@dataclass(frozen=True)
class Trade:
    resting_order_id: int
    price: float
    quantity: float
    timestamp: datetime


class OrderBook:
    """Price-time-priority limit order book for a single symbol.

    Both sides are stored in a `SortedDict` keyed by price (always ascending —
    "best bid" is read via `peekitem(-1)`, "best ask" via `peekitem(0)`; a
    heap was rejected because it can't support O(log n) cancellation or
    ordered level iteration). Each price level is a FIFO `deque`, so orders at
    the same price fill in arrival order (price-time priority).

    A marketable limit order (one that crosses the spread) matches
    immediately against the opposite book before any unfilled remainder
    rests, so the book can never end up crossed (best_bid >= best_ask).
    """

    def __init__(self) -> None:
        self.bids: SortedDict[float, deque[RestingOrder]] = SortedDict()
        self.asks: SortedDict[float, deque[RestingOrder]] = SortedDict()
        self._order_index: dict[int, tuple[Side, float]] = {}
        self._id_counter = itertools.count(1)

    def best_bid(self) -> float | None:
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def add_limit(
        self,
        side: Side,
        price: float,
        quantity: float,
        timestamp: datetime,
        order_id: int | None = None,
    ) -> tuple[int, list[Trade]]:
        """Match immediately against the opposite book up to `price`; rest any
        unfilled remainder. Returns (order_id, trades)."""
        if order_id is None:
            order_id = next(self._id_counter)

        trades, remaining = self._match(side, quantity, timestamp, price_limit=price)

        if remaining > 0:
            resting_book = self.bids if side == "BUY" else self.asks
            resting_book.setdefault(price, deque()).append(
                RestingOrder(order_id, side, price, remaining, timestamp)
            )
            self._order_index[order_id] = (side, price)

        return order_id, trades

    def cancel(self, order_id: int) -> bool:
        info = self._order_index.pop(order_id, None)
        if info is None:
            return False

        side, price = info
        book = self.bids if side == "BUY" else self.asks
        level = book.get(price)
        if level is None:
            return False

        for order in level:
            if order.order_id == order_id:
                level.remove(order)
                break
        if not level:
            del book[price]
        return True

    def market_order(self, side: Side, quantity: float, timestamp: datetime) -> tuple[list[Trade], float]:
        """Walk the opposite book at any price until filled or liquidity runs
        out. Returns (trades, unfilled_quantity)."""
        return self._match(side, quantity, timestamp, price_limit=None)

    def ioc_order(
        self, side: Side, quantity: float, price: float, timestamp: datetime
    ) -> tuple[list[Trade], float]:
        """Fill immediately at `price` or better; any unfilled remainder is
        discarded (never rests on the book). Returns (trades, unfilled_quantity)."""
        return self._match(side, quantity, timestamp, price_limit=price)

    def _match(
        self, side: Side, quantity: float, timestamp: datetime, price_limit: float | None
    ) -> tuple[list[Trade], float]:
        book = self.asks if side == "BUY" else self.bids
        price_levels = list(book.keys())
        if book is self.bids:
            price_levels.reverse()  # walk highest bid first

        trades: list[Trade] = []
        remaining = quantity

        for price in price_levels:
            if remaining <= 0:
                break
            if price_limit is not None:
                if side == "BUY" and price > price_limit:
                    break
                if side == "SELL" and price < price_limit:
                    break

            level = book[price]
            while level and remaining > 0:
                resting = level[0]
                fill_qty = min(resting.quantity, remaining)
                trades.append(Trade(resting.order_id, price, fill_qty, timestamp))
                resting.quantity -= fill_qty
                remaining -= fill_qty
                if resting.quantity <= 0:
                    level.popleft()
                    self._order_index.pop(resting.order_id, None)
            if not level:
                del book[price]

        return trades, remaining
