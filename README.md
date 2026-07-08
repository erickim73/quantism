# QuantSim

A Python quant trading research platform: an event-driven backtesting engine, a limit-order-book exchange simulator, a square-root market impact model, and a Monte Carlo VaR risk engine.

## Status

Work in progress. See `.claude/plans` (implementation plan) for the phased roadmap:

1. Event-driven backtesting engine (portfolio accounting, performance metrics, mean-reversion and momentum strategies)
2. Exchange / order-book simulator (price-time-priority matching, synthetic tick generation)
3. Market impact / microstructure model (square-root impact law, slippage attribution)
4. Monte Carlo VaR risk engine (correlated path simulation, Kupiec backtest)
5. Stretch: portfolio optimization, pairs trading, optional C++ matching-engine port

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Demo results and usage examples will be added here as each phase lands.
