import numpy as np
import pytest
from scipy import stats

from quantsim.risk.monte_carlo_var import (
    block_bootstrap_returns,
    compute_var_cvar,
    portfolio_pnl_distribution,
    regularize_covariance,
    simulate_correlated_returns,
)


def test_regularize_covariance_adds_epsilon_to_diagonal_only():
    covariance = np.array([[0.04, 0.02], [0.02, 0.09]])

    regularized = regularize_covariance(covariance, epsilon=1e-3)

    assert regularized[0, 0] == pytest.approx(0.04 + 1e-3)
    assert regularized[1, 1] == pytest.approx(0.09 + 1e-3)
    assert regularized[0, 1] == pytest.approx(0.02)  # off-diagonal untouched


def test_simulate_correlated_returns_has_expected_shape():
    mean_returns = np.array([0.0005, 0.0003])
    covariance = np.array([[0.0004, 0.0], [0.0, 0.0009]])

    returns = simulate_correlated_returns(mean_returns, covariance, n_scenarios=1000, seed=1)

    assert returns.shape == (1000, 2)


def test_simulate_correlated_returns_is_reproducible_with_seed():
    mean_returns = np.array([0.0])
    covariance = np.array([[0.0004]])

    a = simulate_correlated_returns(mean_returns, covariance, n_scenarios=100, seed=42)
    b = simulate_correlated_returns(mean_returns, covariance, n_scenarios=100, seed=42)

    assert np.array_equal(a, b)


def test_simulate_correlated_returns_recovers_input_correlation_structure():
    mean_returns = np.array([0.0, 0.0])
    true_corr = 0.7
    covariance = np.array([[0.0004, true_corr * 0.02 * 0.03], [true_corr * 0.02 * 0.03, 0.0009]])

    returns = simulate_correlated_returns(mean_returns, covariance, n_scenarios=100_000, seed=7)
    sample_corr = np.corrcoef(returns.T)[0, 1]

    assert sample_corr == pytest.approx(true_corr, abs=0.03)


def test_single_asset_monte_carlo_var_converges_to_analytical_gaussian_var():
    mu, sigma = 0.0, 0.02
    confidence = 0.95

    returns = simulate_correlated_returns(np.array([mu]), np.array([[sigma**2]]), n_scenarios=200_000, seed=3)
    pnl = portfolio_pnl_distribution(returns, weights=np.array([1.0]), portfolio_value=1.0)
    result = compute_var_cvar(pnl, confidence=confidence)

    analytical_var = -(mu + stats.norm.ppf(1 - confidence) * sigma)

    assert result.var == pytest.approx(analytical_var, abs=0.0015)


def test_portfolio_pnl_distribution_scales_by_weights_and_value():
    horizon_returns = np.array([[0.01, -0.02], [0.03, 0.01]])
    weights = np.array([0.5, 0.5])

    pnl = portfolio_pnl_distribution(horizon_returns, weights, portfolio_value=10_000)

    expected = np.array([(0.5 * 0.01 + 0.5 * -0.02) * 10_000, (0.5 * 0.03 + 0.5 * 0.01) * 10_000])
    np.testing.assert_allclose(pnl, expected)


def test_compute_var_cvar_matches_hand_computed_quantile():
    pnl = np.array([-100.0, -50.0, 0.0, 50.0, 100.0, -200.0, -10.0, 20.0, 30.0, -5.0])

    result = compute_var_cvar(pnl, confidence=0.9)

    losses = -pnl
    expected_var = np.quantile(losses, 0.9)
    assert result.var == pytest.approx(expected_var)
    assert result.cvar >= result.var


def test_compute_var_cvar_raises_for_invalid_confidence():
    with pytest.raises(ValueError):
        compute_var_cvar(np.array([1.0, 2.0]), confidence=1.5)


def test_block_bootstrap_returns_has_expected_shape():
    historical = np.random.default_rng(0).normal(0, 0.01, size=(100, 2))

    sampled = block_bootstrap_returns(historical, n_scenarios=500, horizon_days=5, block_size=5, seed=1)

    assert sampled.shape == (500, 2)


def test_block_bootstrap_returns_is_reproducible_with_seed():
    historical = np.random.default_rng(0).normal(0, 0.01, size=(50, 2))

    a = block_bootstrap_returns(historical, n_scenarios=50, horizon_days=3, seed=5)
    b = block_bootstrap_returns(historical, n_scenarios=50, horizon_days=3, seed=5)

    assert np.array_equal(a, b)


def test_block_bootstrap_returns_mean_approximately_matches_scaled_historical_mean():
    rng = np.random.default_rng(0)
    historical = rng.normal(0.001, 0.01, size=(500, 1))

    sampled = block_bootstrap_returns(historical, n_scenarios=10_000, horizon_days=5, block_size=5, seed=2)

    assert sampled.mean() == pytest.approx(5 * historical.mean(), abs=0.015)


def test_block_bootstrap_returns_raises_when_block_size_exceeds_history():
    historical = np.zeros((3, 1))

    with pytest.raises(ValueError):
        block_bootstrap_returns(historical, n_scenarios=10, block_size=5)
