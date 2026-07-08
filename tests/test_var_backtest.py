import numpy as np
import pytest

from quantsim.risk.var_backtest import count_breaches, kupiec_pof_test


def test_count_breaches_with_scalar_var_estimate():
    realized_pnl = np.array([-5.0, -15.0, 3.0, -20.0, 1.0])

    assert count_breaches(realized_pnl, var_estimates=10.0) == 2  # -15 and -20 exceed loss of 10


def test_count_breaches_with_per_period_var_estimates():
    realized_pnl = np.array([-5.0, -15.0, -20.0])
    var_estimates = np.array([10.0, 10.0, 25.0])

    assert count_breaches(realized_pnl, var_estimates) == 1  # only -15 breaches its own 10


def test_kupiec_test_does_not_reject_when_breach_rate_exactly_matches_expected():
    n = 2000
    confidence = 0.95
    n_breaches_expected = 100  # exactly 5% of 2000

    realized_pnl = np.array([-11.0] * n_breaches_expected + [-5.0] * (n - n_breaches_expected))
    result = kupiec_pof_test(realized_pnl, var_estimates=10.0, confidence=confidence)

    assert result.n_breaches == n_breaches_expected
    assert result.breach_rate == pytest.approx(0.05)
    assert result.likelihood_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.reject_null is False


def test_kupiec_test_rejects_when_far_too_many_breaches():
    n = 1000
    realized_pnl = np.array([-11.0] * 200 + [-5.0] * (n - 200))  # 20% breach rate vs 5% expected

    result = kupiec_pof_test(realized_pnl, var_estimates=10.0, confidence=0.95)

    assert result.n_breaches == 200
    assert result.reject_null is True
    assert result.p_value < 0.05


def test_kupiec_test_rejects_when_far_too_few_breaches():
    n = 1000
    realized_pnl = np.array([-11.0] * 2 + [-5.0] * (n - 2))  # 0.2% breach rate vs 5% expected

    result = kupiec_pof_test(realized_pnl, var_estimates=10.0, confidence=0.95)

    assert result.n_breaches == 2
    assert result.reject_null is True


def test_kupiec_test_handles_zero_breaches_without_error():
    realized_pnl = np.array([-5.0] * 100)  # no breaches at all

    result = kupiec_pof_test(realized_pnl, var_estimates=10.0, confidence=0.95)

    assert result.n_breaches == 0
    assert result.breach_rate == 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_kupiec_test_handles_all_breaches_without_error():
    realized_pnl = np.array([-20.0] * 50)  # every period breaches

    result = kupiec_pof_test(realized_pnl, var_estimates=10.0, confidence=0.95)

    assert result.n_breaches == 50
    assert result.breach_rate == 1.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.reject_null is True


def test_kupiec_test_raises_for_empty_input():
    with pytest.raises(ValueError):
        kupiec_pof_test(np.array([]), var_estimates=10.0)
