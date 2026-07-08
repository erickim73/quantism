"""Mean-variance portfolio optimization (Markowitz), with equal-weight and
naive risk-parity baselines for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

DEFAULT_RISK_FREE_RATE = 0.0


def equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def risk_parity_weights(covariance: np.ndarray) -> np.ndarray:
    """Naive ("inverse volatility") risk parity: weight inversely
    proportional to each asset's own volatility, normalized to sum to 1.

    Exact equal risk contribution only holds when assets are uncorrelated;
    this inverse-vol heuristic is nonetheless the standard simple baseline
    used in practice when a full risk-budgeting optimization isn't needed.
    """
    covariance = np.asarray(covariance, dtype=float)
    volatilities = np.sqrt(np.diag(covariance))
    inverse_vol = 1.0 / volatilities
    return inverse_vol / inverse_vol.sum()


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.sqrt(weights @ covariance @ weights))


def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    volatility = portfolio_volatility(weights, covariance)
    if volatility == 0:
        return 0.0
    return (portfolio_return(weights, expected_returns) - risk_free_rate) / volatility


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def evaluate_portfolio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """Evaluate an arbitrary weight vector (e.g. equal-weight or risk-parity)
    on the same return/risk/Sharpe metrics as the optimized portfolios, for
    apples-to-apples comparison."""
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    return OptimizationResult(
        weights=weights,
        expected_return=portfolio_return(weights, expected_returns),
        volatility=portfolio_volatility(weights, covariance),
        sharpe_ratio=portfolio_sharpe(weights, expected_returns, covariance, risk_free_rate),
    )


def _long_only_fully_invested(n_assets: int):
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def maximize_sharpe(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """Long-only, fully-invested Markowitz portfolio maximizing Sharpe ratio."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n_assets = expected_returns.shape[0]

    def negative_sharpe(weights: np.ndarray) -> float:
        return -portfolio_sharpe(weights, expected_returns, covariance, risk_free_rate)

    bounds, constraints = _long_only_fully_invested(n_assets)
    result = minimize(
        negative_sharpe, equal_weights(n_assets), method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not result.success:
        raise RuntimeError(f"Sharpe maximization failed to converge: {result.message}")

    return evaluate_portfolio(result.x, expected_returns, covariance, risk_free_rate)


def minimize_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """Long-only, fully-invested minimum-variance portfolio."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n_assets = expected_returns.shape[0]

    def variance(weights: np.ndarray) -> float:
        return portfolio_volatility(weights, covariance) ** 2

    bounds, constraints = _long_only_fully_invested(n_assets)
    result = minimize(variance, equal_weights(n_assets), method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Variance minimization failed to converge: {result.message}")

    return evaluate_portfolio(result.x, expected_returns, covariance, risk_free_rate)
