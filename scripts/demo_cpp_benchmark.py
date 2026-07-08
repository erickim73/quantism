"""Phase 5 stretch demo: benchmark the pure-Python OrderBook against the
optional C++-accelerated CppOrderBook on an identical stream of synthetic
order flow.

Requires the C++ extension to be installed first:
    pip install ./cpp

Run with: python scripts/demo_cpp_benchmark.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np

from quantsim.exchange.order_book import OrderBook

try:
    from quantsim.exchange.cpp_order_book import CppOrderBook
except ImportError:
    CppOrderBook = None

SEED = 5
N_OPERATIONS = 200_000
MID_PRICE = 100.0
SPREAD = 0.01
T0 = datetime(2024, 1, 1, 9, 30)


def generate_operations(n: int, seed: int) -> list[tuple[str, str, float, float]]:
    """Pre-generate a fixed stream of (kind, side, price, quantity) operations
    so both implementations replay the *exact* same order flow."""
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(n):
        side = "BUY" if rng.random() < 0.5 else "SELL"
        is_market = rng.random() < 0.3
        offset = rng.integers(1, 20) * SPREAD
        price = MID_PRICE - offset if side == "BUY" else MID_PRICE + offset
        quantity = float(rng.integers(1, 20) * 10)
        ops.append(("market" if is_market else "limit", side, price, quantity))
    return ops


def run_benchmark(book, operations) -> float:
    timestamp = T0
    start = time.perf_counter()
    for kind, side, price, quantity in operations:
        timestamp += timedelta(microseconds=1)
        if kind == "market":
            book.market_order(side, quantity, timestamp)
        else:
            book.add_limit(side, price, quantity, timestamp)
    return time.perf_counter() - start


def main() -> None:
    operations = generate_operations(N_OPERATIONS, SEED)

    python_book = OrderBook()
    python_elapsed = run_benchmark(python_book, operations)
    python_throughput = N_OPERATIONS / python_elapsed

    print(f"--- Order book throughput: {N_OPERATIONS:,} operations ---")
    print(f"Pure Python:              {python_elapsed:.3f}s  ({python_throughput:,.0f} ops/sec)")

    if CppOrderBook is None:
        print("\nC++ extension not installed — run `pip install ./cpp` to enable the comparison.")
        return

    cpp_book = CppOrderBook()
    cpp_elapsed = run_benchmark(cpp_book, operations)
    cpp_throughput = N_OPERATIONS / cpp_elapsed
    print(f"C++ (one call per order): {cpp_elapsed:.3f}s  ({cpp_throughput:,.0f} ops/sec)  "
          f"[{python_elapsed / cpp_elapsed:.1f}x]")

    # Calling into C++ once per Python-loop iteration mostly measures pybind11
    # call/marshaling overhead, not the matching engine itself -- a classic
    # FFI benchmarking mistake. Running the whole batch in one C++ call (the
    # hot loop never leaves C++) isolates the actual matching-loop speedup.
    batch_ops = [(kind == "market", side, price, quantity) for kind, side, price, quantity in operations]
    cpp_batch_book = CppOrderBook()
    start = time.perf_counter()
    cpp_batch_book.run_batch(batch_ops)
    cpp_batch_elapsed = time.perf_counter() - start
    cpp_batch_throughput = N_OPERATIONS / cpp_batch_elapsed
    print(f"C++ (batched, one call):  {cpp_batch_elapsed:.3f}s  ({cpp_batch_throughput:,.0f} ops/sec)  "
          f"[{python_elapsed / cpp_batch_elapsed:.1f}x]")

    print(
        "\nLesson: per-call speedup is modest because pybind11's Python<->C++ "
        "boundary crossing dominates a cheap std::map insert; batching the "
        "whole hot loop into one C++ call removes that overhead and reveals "
        "the matching engine's actual speedup."
    )


if __name__ == "__main__":
    main()
