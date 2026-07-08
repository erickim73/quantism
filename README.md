# QuantSim

A Python quant trading research platform built to mirror, at small scale, what
a quant fund's internal research tooling actually looks like: an event-driven
backtesting engine, a limit-order-book exchange simulator, a square-root
market impact model, a Monte Carlo VaR risk engine, and mean-variance
portfolio optimization — composed into one coherent, tested system rather
than four disconnected scripts.

It's deliberately scoped as a demonstration of three things at once:
**quantitative modeling** (the trading/risk logic), **systems engineering**
(the event-driven simulation core), and **software architecture** (how the
pieces compose without leaking abstractions into each other).

## Architecture

```
                    ┌─────────────────┐
                    │   DataHandler   │  (HistoricDataHandler / SyntheticTickDataHandler)
                    │  push MarketEvent│
                    └────────┬────────┘
                             │
                       ┌─────▼─────┐
                       │EventQueue │  timestamp-ordered min-heap
                       │(heapq)    │  (deterministic same-tick tie-break)
                       └─────┬─────┘
                             │ pop, in strict time order
              ┌──────────────┼──────────────────┐
              ▼              ▼                  ▼
        MarketEvent    SignalEvent          FillEvent
              │              │                  │
              ▼              ▼                  ▼
        Strategy.on_data  → OrderEvent   Portfolio.update_fill
        (only sees bars                  (cash, positions,
         strictly before "now")           realized/unrealized P&L)
                             │
                             ▼
                    ExecutionHandler.execute_order
                    ┌────────────────┬─────────────────────┐
                    │ SimpleExecution│ OrderBookExecution   │
                    │ Handler        │ Handler              │
                    │ (fixed bps,    │ (walks a live        │
                    │  next-bar open)│  OrderBook; optional │
                    │                │  MarketImpact wrapper)│
                    └────────────────┴─────────────────────┘
```

`Strategy` and `Portfolio` never know which `ExecutionHandler` is active —
swapping the Phase 1 naive fill model for Phase 2's order-book execution is a
one-line change in the script that wires up the `Backtester` (see
`scripts/demo_tick_replay.py`, which runs the *same* strategy through both).

## What's implemented

| Phase | Module(s) | What it demonstrates |
|---|---|---|
| 1. Backtesting engine | `engine/` | Heap-based event queue, `Strategy` ABC, `Portfolio` accounting (long/short/flip P&L), Sharpe/Sortino/max-drawdown/win-rate/turnover, look-ahead-bias-safe `DataHandler` |
| 2. Exchange simulator | `exchange/order_book.py`, `tick_generator.py`, `execution.py` | Price-time-priority limit order book (`SortedDict` + FIFO deques), synthetic Poisson/Brownian tick generator, order-book-based execution with walk-the-book slippage |
| 3. Market impact | `exchange/market_impact.py` | Square-root impact law calibration, spread/impact/timing slippage attribution |
| 4. Risk engine | `risk/` | Monte Carlo VaR/CVaR (Cholesky or block-bootstrap paths), Kupiec proportion-of-failures backtest |
| 5. Stretch | `strategies/pairs_trading.py`, `optimization/mean_variance.py` | Cointegration-based (Engle-Granger) pairs trading with hedge-ratio-weighted legs, Markowitz mean-variance optimization vs equal-weight/risk-parity baselines |

Two single-asset strategies (`strategies/momentum.py` — MA crossover,
`strategies/mean_reversion.py` — Bollinger/z-score) plug into the same
`Strategy` interface as the pairs-trading strategy.

## Key design decisions

- **Event queue is a min-heap, not a FIFO deque.** Once multiple symbols or
  tick streams are merged into one queue, only a heap guarantees strict
  global time ordering; a `(timestamp, sequence_id, event)` tuple breaks
  same-timestamp ties deterministically without requiring `Event` itself to
  be orderable. See `engine/event_queue.py`.
- **Look-ahead bias is structural, not a convention.** `DataHandler.get_latest_bars`
  only ever returns bars strictly before the in-flight one (tracked via
  `mark_current`), so a strategy cannot accidentally see the future — this
  isn't a rule strategy authors have to remember, it's enforced by the data
  handler's own state machine.
- **Order book uses `SortedDict` + FIFO `deque` levels, not a heap.** A heap
  can't support O(log n) cancellation or ordered level iteration, both of
  which the matching engine needs. A marketable limit order matches
  immediately against the opposite side before any remainder rests, so the
  book can never end up crossed by construction (see the `_match` docstring
  in `exchange/order_book.py`).
- **Covariance matrices are regularized before Cholesky** (`+= 1e-6 * I`) so
  near-singular matrices (e.g. two highly correlated assets) never crash the
  Monte Carlo VaR path simulation.
- **The synthetic tick generator has two modes for a reason.**
  `generate_ohlcv` pre-generates a whole stream (fine for a fixed-cost
  execution model that only reads OHLCV values); `SyntheticTickDataHandler`
  steps the generator incrementally so `order_book` state stays in sync with
  "now" during replay — reusing the pre-generated book for live execution
  would silently fill every order against the *final* tick's liquidity. See
  the docstrings in `exchange/tick_generator.py` for the full reasoning.
- **The C++/pybind11 matching-engine port is optional and separately
  installed, not a core dependency.** The pure-Python order book already
  replays comfortably fast for this project's data volumes, so nothing in
  `quantsim` requires the C++ extension — `cpp/` is its own installable
  package (`pip install ./cpp`, built with `scikit-build-core` + CMake +
  MSVC) exercising the same interface as `exchange/order_book.py`, verified
  for behavioral parity in `tests/test_cpp_order_book.py` (same test bodies
  run against both implementations). It exists as a systems-engineering
  exercise in FFI design, not because the project needed the speed.
  `scripts/demo_cpp_benchmark.py` reports an honest result rather than a
  flattering one: calling into C++ once per order is only ~1.2x faster than
  pure Python, because pybind11's Python↔C++ boundary crossing dominates a
  cheap `std::map` insert — a classic FFI benchmarking mistake. Batching an
  entire operation list into one C++ call via `run_batch` (so the hot loop
  never leaves C++) reveals the matching engine's actual throughput: ~8.3x.

## Quant methodology references

- **Square-root market impact law**: `I(Q) = Y · σ · sqrt(Q / ADV)`, Y
  typically calibrated in [0.5, 1.0] for US equities — see Almgren, Thum,
  Hauptmann & Li (2005), *Direct Estimation of Equity Market Impact*.
- **Kupiec proportion-of-failures test** (1995): likelihood-ratio test for
  whether a VaR model's observed breach rate matches its stated confidence
  level; asymptotically χ²(1) under the null of correct calibration.
- **Engle-Granger cointegration test**: tests for a stable long-run
  equilibrium relationship between two price series, the standard
  prerequisite check before trading a pairs/stat-arb spread.
- **Markowitz mean-variance optimization**: long-only, fully-invested
  max-Sharpe and min-variance portfolios via `scipy.optimize.minimize`
  (SLSQP), validated in tests against closed-form solutions for the
  uncorrelated-asset special case.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Dependencies: numpy, pandas, scipy, matplotlib, yfinance, sortedcontainers,
statsmodels (see `pyproject.toml`).

## Running the demos

Each script is self-contained and prints its results (several also save a
plot to `output/`):

| Script | Demonstrates |
|---|---|
| `scripts/demo_ma_crossover.py` | Phase 1: MA-crossover backtest on real AAPL daily data via yfinance, equity curve plot + trade log CSV + Sharpe/Sortino/drawdown/turnover |
| `scripts/demo_tick_replay.py` | Phase 2: the same strategy on an identical synthetic tick path, comparing naive fixed-bps fills vs real order-book execution |
| `scripts/demo_market_impact.py` | Phase 3: cost-vs-size curve calibration and a worked slippage-attribution example |
| `scripts/demo_var.py` | Phase 4: Monte Carlo VaR/CVaR for a 2-asset portfolio + Kupiec backtest verdict |
| `scripts/demo_pairs_trading.py` | Phase 5: cointegration check + pairs-trading backtest on a synthetic cointegrated pair |
| `scripts/demo_optimization.py` | Phase 5: max-Sharpe / min-variance vs equal-weight / risk-parity, plotted against random portfolios |
| `scripts/demo_cpp_benchmark.py` | Phase 5 (optional): pure-Python vs C++ order book throughput — requires `pip install ./cpp` first, otherwise prints a note and exits |

`demo_ma_crossover.py` is the only script that hits the network (yfinance);
all others use synthetic data generated in-process, so they run offline.

Sample output (`demo_var.py`):

```
--- Monte Carlo VaR / CVaR (1-day horizon, calibrated from 500 historical days) ---
95% VaR:  $   23,188.16
95% CVaR: $   29,093.23
99% VaR:  $   32,738.59
99% CVaR: $   37,677.57

--- Kupiec proportion-of-failures backtest (95% VaR, 250 out-of-sample days) ---
Breaches: 15 (expected ~12.5)
Breach rate: 0.0600 (expected 0.0500)
P-value: 0.4812
Verdict: PASSED (model well-calibrated)
```

## Testing

```bash
pytest                       # 163 tests, runs in under 2 seconds
pytest --cov=quantsim --cov-report=term-missing
```

Tests favor behavioral coverage over line-count coverage: portfolio
accounting is tested through long/short/partial-close/position-flip P&L
scenarios, the order book through price-time-priority and multi-level
walk-the-book cases, and the Monte Carlo VaR engine against closed-form
analytical solutions where one exists (single-asset Gaussian VaR, 2-asset
uncorrelated tangency/min-variance portfolios) rather than only against
its own output.

## Project structure

```
quantsim/
├── engine/          # event queue, Strategy ABC, Portfolio, metrics, naive execution, Backtester
├── exchange/         # order book, synthetic tick generator, order-book execution, market impact
├── risk/             # Monte Carlo VaR/CVaR, Kupiec backtest
├── strategies/       # momentum, mean-reversion, pairs trading
├── optimization/      # Markowitz mean-variance
└── data/             # yfinance loader + CSV cache, HistoricDataHandler
scripts/              # runnable demos (one per phase, see table above)
tests/                # 163 tests, one file per module
cpp/                  # optional C++ matching-engine extension (pip install ./cpp)
├── src/order_book.cpp
├── CMakeLists.txt
└── pyproject.toml    # scikit-build-core + pybind11 build backend
```

## Known limitations

- No real limit-order-book tick data (LOBSTER) is used — the order book is
  driven by a synthetic Poisson/Brownian generator instead, by deliberate
  choice (see design decisions above), not by omission.
- The synthetic order book's depth is uniform across levels, which produces
  a roughly *linear* (not the classic concave square-root-shaped) mechanical
  walk-the-book cost curve; the empirical square-root law comes from real
  markets' non-uniform, typically decaying depth profiles. `demo_market_impact.py`
  calibrates a Y coefficient to fit the model to this book's cost curve
  anyway, which is an honest illustration of *why* real books look different
  from a toy uniform one, not a claim that they match exactly.
- `demo_ma_crossover.py` is the only demo that depends on fetching real data
  from yfinance; this repo's own dev environment had no network access to
  verify that specific script end-to-end — run it yourself to confirm.
- The C++ matching-engine extension is not installed by default (`pip install
  ./cpp`, requires a C++ compiler); `demo_cpp_benchmark.py` degrades
  gracefully to a Python-only report if it's absent, and
  `tests/test_cpp_order_book.py` is skipped rather than failed in that case.
