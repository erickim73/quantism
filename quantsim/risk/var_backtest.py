"""Kupiec (1995) proportion-of-failures (POF) backtest for a VaR model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class KupiecTestResult:
    n_observations: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    likelihood_ratio: float
    p_value: float
    reject_null: bool


def count_breaches(realized_pnl: np.ndarray, var_estimates: np.ndarray) -> int:
    """A breach occurs when the realized loss (negative P&L) exceeds the VaR
    estimate for that period. `var_estimates` are positive loss magnitudes,
    either one per observation or a single scalar applied to every period."""
    realized_pnl = np.asarray(realized_pnl, dtype=float)
    var_estimates = np.broadcast_to(np.asarray(var_estimates, dtype=float), realized_pnl.shape)
    losses = -realized_pnl
    return int(np.sum(losses > var_estimates))


def kupiec_pof_test(
    realized_pnl: np.ndarray,
    var_estimates: np.ndarray,
    confidence: float = 0.95,
    significance: float = 0.05,
) -> KupiecTestResult:
    """Is the observed VaR breach rate consistent with the model's stated
    confidence level?

    Under the null hypothesis (a correctly calibrated VaR model), the
    likelihood-ratio statistic is asymptotically chi-square distributed with
    1 degree of freedom. Rejecting the null means the breach rate is
    statistically inconsistent with the claimed confidence level — either too
    many breaches (VaR underestimates risk) or suspiciously few (VaR is overly
    conservative).
    """
    realized_pnl = np.asarray(realized_pnl, dtype=float)
    n = realized_pnl.shape[0]
    if n == 0:
        raise ValueError("realized_pnl must be non-empty")

    n_breaches = count_breaches(realized_pnl, var_estimates)
    p_expected = 1 - confidence
    p_observed = n_breaches / n

    likelihood_ratio_stat = -2 * _log_likelihood_ratio(n, n_breaches, p_expected, p_observed)
    p_value = float(1 - stats.chi2.cdf(likelihood_ratio_stat, df=1))

    return KupiecTestResult(
        n_observations=n,
        n_breaches=n_breaches,
        breach_rate=p_observed,
        expected_rate=p_expected,
        likelihood_ratio=likelihood_ratio_stat,
        p_value=p_value,
        reject_null=p_value < significance,
    )


def _log_likelihood_ratio(n: int, x: int, p_expected: float, p_observed: float) -> float:
    """log[ L(p_expected) / L(p_observed) ] for the binomial breach-count
    likelihood, treating the degenerate 0*log(0) case (x=0 or x=n) as 0."""

    def log_likelihood(p: float, k: int, m: int) -> float:
        term = 0.0
        if k > 0:
            term += k * np.log(p)
        if m - k > 0:
            term += (m - k) * np.log(1 - p)
        return term

    log_l_expected = log_likelihood(p_expected, x, n)
    log_l_observed = log_likelihood(p_observed, x, n) if 0 < p_observed < 1 else 0.0
    return log_l_expected - log_l_observed
