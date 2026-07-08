"""Phase 5 demo: Markowitz mean-variance optimization vs equal-weight and
naive risk-parity baselines, plotting the achievable risk/return frontier.

Uses assumed (illustrative) expected returns and a covariance matrix for a
handful of assets — this sandbox has no network access to estimate these from
real historical data.

Run with: python scripts/demo_optimization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantsim.optimization.mean_variance import (
    equal_weights,
    evaluate_portfolio,
    maximize_sharpe,
    minimize_variance,
    risk_parity_weights,
)

ASSET_NAMES = ["Equity_A", "Equity_B", "Bond_C", "Commodity_D"]
EXPECTED_RETURNS = np.array([0.10, 0.14, 0.04, 0.07])
COVARIANCE = np.array(
    [
        [0.040, 0.020, 0.002, 0.010],
        [0.020, 0.090, 0.001, 0.015],
        [0.002, 0.001, 0.003, -0.002],
        [0.010, 0.015, -0.002, 0.030],
    ]
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def random_long_only_portfolios(n_portfolios: int, n_assets: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_assets), size=n_portfolios)


def main() -> None:
    max_sharpe = maximize_sharpe(EXPECTED_RETURNS, COVARIANCE)
    min_variance = minimize_variance(EXPECTED_RETURNS, COVARIANCE)
    equal = evaluate_portfolio(equal_weights(len(ASSET_NAMES)), EXPECTED_RETURNS, COVARIANCE)
    risk_parity = evaluate_portfolio(risk_parity_weights(COVARIANCE), EXPECTED_RETURNS, COVARIANCE)

    print("--- Portfolio comparison ---")
    print(f"{'Portfolio':<20}{'Return':>10}{'Vol':>10}{'Sharpe':>10}")
    for label, result in (
        ("Max Sharpe", max_sharpe),
        ("Min Variance", min_variance),
        ("Equal Weight", equal),
        ("Risk Parity", risk_parity),
    ):
        print(f"{label:<20}{result.expected_return:>10.4f}{result.volatility:>10.4f}{result.sharpe_ratio:>10.4f}")

    print("\n--- Max-Sharpe weights ---")
    for name, weight in zip(ASSET_NAMES, max_sharpe.weights):
        print(f"  {name}: {weight:.4f}")

    random_weights = random_long_only_portfolios(2000, len(ASSET_NAMES))
    random_returns = random_weights @ EXPECTED_RETURNS
    random_vols = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, COVARIANCE, random_weights))

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(random_vols, random_returns, s=6, alpha=0.3, label="Random long-only portfolios")
    plt.scatter(max_sharpe.volatility, max_sharpe.expected_return, c="red", marker="*", s=200, label="Max Sharpe")
    plt.scatter(min_variance.volatility, min_variance.expected_return, c="blue", marker="*", s=200, label="Min Variance")
    plt.scatter(equal.volatility, equal.expected_return, c="green", marker="D", s=80, label="Equal Weight")
    plt.scatter(risk_parity.volatility, risk_parity.expected_return, c="orange", marker="D", s=80, label="Risk Parity")
    plt.xlabel("Volatility (annualized)")
    plt.ylabel("Expected return (annualized)")
    plt.title("Efficient frontier: optimized portfolios vs baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficient_frontier.png")
    print(f"\nSaved efficient frontier plot to {OUTPUT_DIR / 'efficient_frontier.png'}")


if __name__ == "__main__":
    main()
