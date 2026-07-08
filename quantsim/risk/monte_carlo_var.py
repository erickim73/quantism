"""Monte Carlo Value-at-Risk (VaR) and Conditional VaR (CVaR / expected
shortfall) via correlated multi-asset return path simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_REGULARIZATION = 1e-6


def regularize_covariance(covariance: np.ndarray, epsilon: float = DEFAULT_REGULARIZATION) -> np.ndarray:
    """Nudge a covariance matrix to be positive-definite by adding a small
    multiple of the identity to the diagonal, so Cholesky never fails on a
    near-singular matrix (e.g. two nearly perfectly correlated assets)."""
    covariance = np.asarray(covariance, dtype=float)
    return covariance + epsilon * np.eye(covariance.shape[0])


def simulate_correlated_returns(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    n_scenarios: int,
    horizon_days: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate `n_scenarios` paths of correlated returns per asset, summed
    over `horizon_days`, via Cholesky decomposition of the (regularized)
    covariance matrix applied to independent standard-normal draws.

    Returns an (n_scenarios, n_assets) array of horizon returns.
    """
    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance = regularize_covariance(covariance)
    n_assets = mean_returns.shape[0]

    rng = np.random.default_rng(seed)
    cholesky = np.linalg.cholesky(covariance)

    horizon_returns = np.zeros((n_scenarios, n_assets))
    for _ in range(horizon_days):
        independent_draws = rng.standard_normal((n_scenarios, n_assets))
        correlated_draws = independent_draws @ cholesky.T
        horizon_returns += mean_returns + correlated_draws
    return horizon_returns


def block_bootstrap_returns(
    historical_returns: np.ndarray,
    n_scenarios: int,
    horizon_days: int = 1,
    block_size: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate horizon returns via block bootstrap of historical multi-asset
    returns. Sampling contiguous blocks (rather than i.i.d. days) preserves
    cross-asset correlation and short-term autocorrelation, and inherits
    whatever fat tails are present in the historical data instead of assuming
    Gaussian returns.

    `historical_returns` is a (n_days, n_assets) array. Returns an
    (n_scenarios, n_assets) array of horizon returns.
    """
    historical_returns = np.asarray(historical_returns, dtype=float)
    n_days, n_assets = historical_returns.shape
    if block_size > n_days:
        raise ValueError("block_size cannot exceed the number of historical days")

    rng = np.random.default_rng(seed)
    horizon_returns = np.zeros((n_scenarios, n_assets))

    for scenario in range(n_scenarios):
        days_filled = 0
        while days_filled < horizon_days:
            start = rng.integers(0, n_days - block_size + 1)
            block = historical_returns[start : start + block_size]
            take = min(block_size, horizon_days - days_filled)
            horizon_returns[scenario] += block[:take].sum(axis=0)
            days_filled += take

    return horizon_returns


def portfolio_pnl_distribution(
    horizon_returns: np.ndarray, weights: np.ndarray, portfolio_value: float
) -> np.ndarray:
    """Convert simulated per-asset horizon returns into simulated portfolio
    dollar P&L for a given weight vector and current portfolio value."""
    weights = np.asarray(weights, dtype=float)
    portfolio_returns = np.asarray(horizon_returns, dtype=float) @ weights
    return portfolio_returns * portfolio_value


@dataclass(frozen=True)
class VarResult:
    var: float
    cvar: float
    confidence: float


def compute_var_cvar(pnl_distribution: np.ndarray, confidence: float = 0.95) -> VarResult:
    """VaR/CVaR at `confidence` from a simulated P&L distribution.

    VaR is reported as a positive loss magnitude: losses exceed it only
    (1 - confidence) of the time. CVaR (expected shortfall) is the average
    loss in that tail, and is always >= VaR.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    losses = -np.asarray(pnl_distribution, dtype=float)
    var = float(np.quantile(losses, confidence))
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean()) if tail_losses.size > 0 else var
    return VarResult(var=var, cvar=cvar, confidence=confidence)
