import numpy as np
import pytest

from src.portfolio_optimizer import PortfolioOptimizer


def _diagonal_case():
    """3 assets with zero factor loadings (so Sigma reduces exactly to diag(specific_vars)) and
    variances [0.01, 0.04, 0.09] (vol 10%/20%/30%) — a case with known closed-form answers for
    both min-variance (inverse-variance weighting) and risk-parity (inverse-vol weighting)."""
    n_factors = 2
    loadings = np.zeros((3, n_factors))
    factor_cov = np.eye(n_factors) * 0.05  # irrelevant since loadings are all zero
    specific_vars = np.array([0.01, 0.04, 0.09])
    return loadings, factor_cov, specific_vars


def test_min_variance_weights_matches_inverse_variance_closed_form():
    loadings, factor_cov, specific_vars = _diagonal_case()
    w = PortfolioOptimizer.min_variance_weights(loadings, factor_cov, specific_vars)

    inv_var = 1.0 / specific_vars
    expected = inv_var / inv_var.sum()

    np.testing.assert_allclose(w, expected, atol=1e-3)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_risk_parity_weights_matches_inverse_vol_closed_form():
    loadings, factor_cov, specific_vars = _diagonal_case()
    w = PortfolioOptimizer.risk_parity_weights(loadings, factor_cov, specific_vars)

    inv_vol = 1.0 / np.sqrt(specific_vars)
    expected = inv_vol / inv_vol.sum()

    np.testing.assert_allclose(w, expected, atol=1e-3)
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_equalizes_risk_contributions():
    loadings, factor_cov, specific_vars = _diagonal_case()
    w = PortfolioOptimizer.risk_parity_weights(loadings, factor_cov, specific_vars)
    Sigma = PortfolioOptimizer.build_covariance(loadings, factor_cov, specific_vars)
    rc = w * (Sigma @ w)
    assert np.allclose(rc, rc.mean(), atol=1e-4)


def test_min_variance_respects_max_weight_cap():
    loadings, factor_cov, specific_vars = _diagonal_case()
    # Unconstrained min-variance would put ~73% into the lowest-vol asset; cap it at 50%.
    w = PortfolioOptimizer.min_variance_weights(loadings, factor_cov, specific_vars, max_weight=0.5)
    assert w.max() <= 0.5 + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_lower_variance_than_equal_weight():
    loadings, factor_cov, specific_vars = _diagonal_case()
    Sigma = PortfolioOptimizer.build_covariance(loadings, factor_cov, specific_vars)
    w_opt = PortfolioOptimizer.min_variance_weights(loadings, factor_cov, specific_vars)
    w_eq = np.full(3, 1.0 / 3)
    assert (w_opt @ Sigma @ w_opt) <= (w_eq @ Sigma @ w_eq) + 1e-9


def test_infeasible_max_weight_raises():
    loadings, factor_cov, specific_vars = _diagonal_case()
    with pytest.raises(ValueError):
        PortfolioOptimizer.min_variance_weights(loadings, factor_cov, specific_vars, max_weight=0.2)
    with pytest.raises(ValueError):
        PortfolioOptimizer.risk_parity_weights(loadings, factor_cov, specific_vars, max_weight=0.2)
