import numpy as np
import pytest

from quantsim.optimization.mean_variance import (
    equal_weights,
    evaluate_portfolio,
    maximize_sharpe,
    minimize_variance,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
    risk_parity_weights,
)


def test_equal_weights_sums_to_one_and_splits_evenly():
    weights = equal_weights(4)

    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25])


def test_risk_parity_weights_are_inversely_proportional_to_volatility():
    covariance = np.array([[0.01, 0.0], [0.0, 0.04]])  # vols: 0.1, 0.2

    weights = risk_parity_weights(covariance)

    np.testing.assert_allclose(weights, [2 / 3, 1 / 3], rtol=1e-6)
    assert weights.sum() == pytest.approx(1.0)


def test_portfolio_return_is_weighted_average():
    assert portfolio_return(np.array([0.5, 0.5]), np.array([0.10, 0.20])) == pytest.approx(0.15)


def test_portfolio_volatility_matches_quadratic_form():
    weights = np.array([0.5, 0.5])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    expected = np.sqrt(weights @ covariance @ weights)
    assert portfolio_volatility(weights, covariance) == pytest.approx(expected)


def test_portfolio_sharpe_is_zero_when_volatility_is_zero():
    weights = np.array([1.0])
    assert portfolio_sharpe(weights, np.array([0.05]), np.array([[0.0]])) == 0.0


def test_evaluate_portfolio_matches_hand_computed_metrics():
    weights = np.array([0.5, 0.5])
    expected_returns = np.array([0.10, 0.20])
    covariance = np.array([[0.04, 0.0], [0.0, 0.09]])

    result = evaluate_portfolio(weights, expected_returns, covariance)

    assert result.expected_return == pytest.approx(0.15)
    assert result.volatility == pytest.approx(np.sqrt(weights @ covariance @ weights))
    assert result.sharpe_ratio == pytest.approx(result.expected_return / result.volatility)


def test_maximize_sharpe_matches_closed_form_tangency_portfolio_for_diagonal_covariance():
    expected_returns = np.array([0.10, 0.05])
    covariance = np.array([[0.04, 0.0], [0.0, 0.01]])  # diagonal -> no correlation

    result = maximize_sharpe(expected_returns, covariance)

    raw = expected_returns / np.diag(covariance)
    closed_form_weights = raw / raw.sum()
    np.testing.assert_allclose(result.weights, closed_form_weights, atol=0.01)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-9).all()


def test_minimize_variance_matches_closed_form_inverse_variance_weights():
    expected_returns = np.array([0.10, 0.05])
    covariance = np.array([[0.04, 0.0], [0.0, 0.01]])

    result = minimize_variance(expected_returns, covariance)

    raw = 1.0 / np.diag(covariance)
    closed_form_weights = raw / raw.sum()
    np.testing.assert_allclose(result.weights, closed_form_weights, atol=0.01)


def test_minimize_variance_never_worse_than_equal_weight_baseline():
    expected_returns = np.array([0.08, 0.12, 0.05])
    covariance = np.array([[0.05, 0.02, 0.01], [0.02, 0.09, 0.015], [0.01, 0.015, 0.03]])

    optimized = minimize_variance(expected_returns, covariance)
    baseline = evaluate_portfolio(equal_weights(3), expected_returns, covariance)

    assert optimized.volatility <= baseline.volatility + 1e-9


def test_maximize_sharpe_never_worse_than_equal_weight_baseline():
    expected_returns = np.array([0.08, 0.12, 0.05])
    covariance = np.array([[0.05, 0.02, 0.01], [0.02, 0.09, 0.015], [0.01, 0.015, 0.03]])

    optimized = maximize_sharpe(expected_returns, covariance)
    baseline = evaluate_portfolio(equal_weights(3), expected_returns, covariance)

    assert optimized.sharpe_ratio >= baseline.sharpe_ratio - 1e-9


def test_optimizers_respect_long_only_and_fully_invested_constraints():
    expected_returns = np.array([0.08, 0.12, 0.05])
    covariance = np.array([[0.05, 0.02, 0.01], [0.02, 0.09, 0.015], [0.01, 0.015, 0.03]])

    for result in (maximize_sharpe(expected_returns, covariance), minimize_variance(expected_returns, covariance)):
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (result.weights >= -1e-9).all()
        assert (result.weights <= 1 + 1e-9).all()
