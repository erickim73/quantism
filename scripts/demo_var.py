"""Phase 4 demo: Monte Carlo VaR/CVaR for a 2-asset portfolio, calibrated from
historical returns and backtested against realized out-of-sample P&L via the
Kupiec proportion-of-failures test.

Run with: python scripts/demo_var.py
"""

from __future__ import annotations

import numpy as np

from quantsim.risk.monte_carlo_var import compute_var_cvar, portfolio_pnl_distribution, simulate_correlated_returns
from quantsim.risk.var_backtest import kupiec_pof_test

SEED = 11
N_HISTORICAL_DAYS = 500
N_BACKTEST_DAYS = 250
N_SCENARIOS = 100_000
PORTFOLIO_VALUE = 1_000_000.0
WEIGHTS = np.array([0.6, 0.4])

# Stand-in for two correlated equities' daily return process, used to
# generate both the historical calibration sample and the out-of-sample
# backtest days (this sandbox has no network access to fetch real returns —
# see README for the yfinance-backed path).
TRUE_MEAN = np.array([0.0004, 0.0003])
TRUE_COVARIANCE = np.array([[0.00025, 0.00012], [0.00012, 0.00035]])


def main() -> None:
    rng = np.random.default_rng(SEED)

    historical_returns = rng.multivariate_normal(TRUE_MEAN, TRUE_COVARIANCE, size=N_HISTORICAL_DAYS)
    sample_mean = historical_returns.mean(axis=0)
    sample_cov = np.cov(historical_returns.T)

    simulated_returns = simulate_correlated_returns(sample_mean, sample_cov, N_SCENARIOS, seed=SEED + 1)
    simulated_pnl = portfolio_pnl_distribution(simulated_returns, WEIGHTS, PORTFOLIO_VALUE)

    print("--- Monte Carlo VaR / CVaR (1-day horizon, calibrated from 500 historical days) ---")
    for confidence in (0.95, 0.99):
        result = compute_var_cvar(simulated_pnl, confidence=confidence)
        print(f"{int(confidence * 100)}% VaR:  ${result.var:>12,.2f}")
        print(f"{int(confidence * 100)}% CVaR: ${result.cvar:>12,.2f}")

    var_95 = compute_var_cvar(simulated_pnl, confidence=0.95).var

    backtest_returns = rng.multivariate_normal(TRUE_MEAN, TRUE_COVARIANCE, size=N_BACKTEST_DAYS)
    realized_pnl = backtest_returns @ WEIGHTS * PORTFOLIO_VALUE

    kupiec_result = kupiec_pof_test(realized_pnl, var_estimates=var_95, confidence=0.95)

    print("\n--- Kupiec proportion-of-failures backtest (95% VaR, 250 out-of-sample days) ---")
    print(f"Observations: {kupiec_result.n_observations}")
    expected_breaches = kupiec_result.expected_rate * kupiec_result.n_observations
    print(f"Breaches: {kupiec_result.n_breaches} (expected ~{expected_breaches:.1f})")
    print(f"Breach rate: {kupiec_result.breach_rate:.4f} (expected {kupiec_result.expected_rate:.4f})")
    print(f"Likelihood ratio: {kupiec_result.likelihood_ratio:.4f}")
    print(f"P-value: {kupiec_result.p_value:.4f}")
    verdict = "REJECTED (model miscalibrated)" if kupiec_result.reject_null else "PASSED (model well-calibrated)"
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
