"""Optional C++-accelerated order book backend.

Requires the sibling `quantsim-matching-engine` package to be installed
(`pip install ./cpp` from the repo root) — a deliberately separate,
optional install so the core `quantsim` package stays pure Python and
installable without a C++ toolchain. See the README's "Key design
decisions" section for why this exists at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

try:
    import quantsim_matching_engine as _native
except ImportError as exc:
    raise ImportError(
        "CppOrderBook requires the optional C++ extension. Install it with "
        "`pip install ./cpp` from the quantsim repo root (requires a C++ "
        "compiler)."
    ) from exc

from quantsim.exchange.order_book import Trade

Side = Literal["BUY", "SELL"]


class CppOrderBook:
    """Drop-in replacement for `quantsim.exchange.order_book.OrderBook`,
    backed by the compiled C++ matching engine.

    Same public interface and semantics — verified in
    tests/test_cpp_order_book.py against the pure-Python implementation with
    the same test bodies run against both. Only the matching loop itself
    runs in C++.
    """

    def __init__(self) -> None:
        self._impl = _native.OrderBook()

    def best_bid(self) -> float | None:
        return self._impl.best_bid()

    def best_ask(self) -> float | None:
        return self._impl.best_ask()

    def add_limit(
        self,
        side: Side,
        price: float,
        quantity: float,
        timestamp: datetime,
        order_id: int | None = None,
    ) -> tuple[int, list[Trade]]:
        order_id_out, native_trades = self._impl.add_limit(side, price, quantity, timestamp, order_id)
        return order_id_out, [_to_trade(t) for t in native_trades]

    def cancel(self, order_id: int) -> bool:
        return self._impl.cancel(order_id)

    def market_order(self, side: Side, quantity: float, timestamp: datetime) -> tuple[list[Trade], float]:
        native_trades, unfilled = self._impl.market_order(side, quantity, timestamp)
        return [_to_trade(t) for t in native_trades], unfilled

    def ioc_order(self, side: Side, quantity: float, price: float, timestamp: datetime) -> tuple[list[Trade], float]:
        native_trades, unfilled = self._impl.ioc_order(side, quantity, price, timestamp)
        return [_to_trade(t) for t in native_trades], unfilled

    def run_batch(self, operations: list[tuple[bool, Side, float, float]]) -> int:
        """Run a batch of (is_market, side, price, quantity) operations
        entirely within C++ in a single call, avoiding a Python/C++ boundary
        crossing per operation. Returns the total number of trades executed.

        A benchmark that calls `market_order`/`add_limit` once per Python
        loop iteration mostly measures pybind11 call overhead, not the
        matching logic itself — see scripts/demo_cpp_benchmark.py, which
        uses this method for a fair throughput comparison.
        """
        return self._impl.run_batch(operations)


def _to_trade(native_trade) -> Trade:
    return Trade(
        resting_order_id=native_trade.resting_order_id,
        price=native_trade.price,
        quantity=native_trade.quantity,
        timestamp=native_trade.timestamp,
    )
