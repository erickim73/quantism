from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Event:
    timestamp: datetime


@dataclass(frozen=True)
class MarketEvent(Event):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SignalEvent(Event):
    symbol: str
    direction: Literal["LONG", "SHORT", "EXIT"]
    strength: float = 1.0


@dataclass(frozen=True)
class OrderEvent(Event):
    symbol: str
    order_type: Literal["MARKET", "LIMIT", "IOC"]
    direction: Literal["BUY", "SELL"]
    quantity: float
    limit_price: float | None = None


@dataclass(frozen=True)
class FillEvent(Event):
    symbol: str
    direction: Literal["BUY", "SELL"]
    quantity: float
    fill_price: float
    commission: float
    slippage: float = 0.0


class EventQueue:
    """Timestamp-ordered min-heap of events with deterministic tie-breaking.

    Same-timestamp events are returned in insertion order (FIFO), which matters
    once multiple symbol/tick streams are merged into a single queue.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[datetime, int, Event]] = []
        self._counter = itertools.count()

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.timestamp, next(self._counter), event))

    def pop(self) -> Event:
        if not self._heap:
            raise IndexError("pop from an empty EventQueue")
        _, _, event = heapq.heappop(self._heap)
        return event

    def peek(self) -> Event:
        if not self._heap:
            raise IndexError("peek from an empty EventQueue")
        _, _, event = self._heap[0]
        return event

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
