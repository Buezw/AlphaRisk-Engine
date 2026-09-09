import pytest
import numpy as np
from src.factor_model import FactorExposureEngine
from src.risk_decomposition import RiskDecompositionEngine

def test_ols_analytical_solution():
    """Verifies that the hand-rolled OLS solver can precisely recover known, synthetically constructed true parameters"""
    np.random.seed(42)
    N, K = 120, 3
    X = np.random.randn(N, K)
    X[:, 0] = 1.0  # intercept term
    true_betas = np.array([0.05, 1.2, -0.8])
    
    noise = np.random.normal(0, 0.01, size=N)
    y = np.dot(X, true_betas) + noise
    
    calc_betas, s_var, r2 = FactorExposureEngine.solve_ols(X, y)
    
    np.testing.assert_allclose(calc_betas, true_betas, atol=1e-2)
    assert 0.9 <= r2 <= 1.0

def test_euler_risk_identity():
    """Mathematical identity test: factor variance + specific variance must exactly equal total variance"""
    weights = np.array([0.4, 0.6])
    loadings = np.array([
        [1.1, 0.3],
        [0.9, -0.2]
    ])
    factor_cov = np.array([
        [0.04, 0.01],
        [0.01, 0.02]
    ])
    specific_vars = np.array([0.005, 0.008])

    res = RiskDecompositionEngine.decompose_risk(weights, loadings, factor_cov, specific_vars)

    assert np.isclose(res["factor_variance"] + res["specific_variance"], res["total_variance"])


def test_risk_contribution_euler_identity():
    """Sum of per-holding Euler risk contributions must exactly equal total annualized vol,
    and their percentage shares must sum to 1 — the additive-decomposition guarantee that makes
    compute_risk_contributions() meaningful as a "who's driving portfolio risk" breakdown."""
    weights = np.array([0.4, 0.6])
    loadings = np.array([
        [1.1, 0.3],
        [0.9, -0.2]
    ])
    factor_cov = np.array([
        [0.04, 0.01],
        [0.01, 0.02]
    ])
    specific_vars = np.array([0.005, 0.008])

    res = RiskDecompositionEngine.compute_risk_contributions(weights, loadings, factor_cov, specific_vars)

    assert np.isclose(np.sum(res["risk_contribution"]), res["total_annualized_vol"])
    assert np.isclose(np.sum(res["risk_contribution_pct"]), 1.0)


def test_risk_contribution_reflects_correlation_not_just_beta():
    """Two holdings with identical standalone loadings should get different risk contributions
    when one is more correlated with the rest of the portfolio than the other — the whole point
    of using the full covariance matrix Σ = B Σ_F Bᵀ + D instead of isolated factor exposures."""
    # Three assets, same single-factor loading (1.0), but asset C carries much higher specific
    # (idiosyncratic) variance that is uncorrelated with A and B, and is heavily underweighted.
    weights = np.array([0.45, 0.45, 0.10])
    loadings = np.array([
        [1.0],
        [1.0],
        [1.0],
    ])
    factor_cov = np.array([[0.02]])
    specific_vars = np.array([0.001, 0.001, 0.05])

    res = RiskDecompositionEngine.compute_risk_contributions(
        weights, loadings, factor_cov, specific_vars, symbols=["A", "B", "C"]
    )

    # A and B are identical in every respect and equally weighted -> identical risk contributions
    assert np.isclose(res["risk_contribution"]["A"], res["risk_contribution"]["B"])
    # C's risk share should exceed its capital share, since its outsized idiosyncratic variance
    # isn't diversified away by the other two holdings
    assert res["risk_contribution_pct"]["C"] > weights[2]


def test_correlation_matrix_properties():
    """Basic sanity properties any valid correlation matrix must satisfy: unit diagonal,
    symmetry, and values bounded in [-1, 1]."""
    loadings = np.array([
        [1.1, 0.3],
        [0.9, -0.2],
        [1.0, 0.0],
    ])
    factor_cov = np.array([
        [0.04, 0.01],
        [0.01, 0.02]
    ])
    specific_vars = np.array([0.005, 0.008, 0.02])

    res = RiskDecompositionEngine.compute_correlation_matrix(loadings, factor_cov, specific_vars, symbols=["A", "B", "C"])
    m = np.array(res["correlation_matrix"])

    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)
    assert np.all(m <= 1.0 + 1e-9) and np.all(m >= -1.0 - 1e-9)


def test_correlation_matrix_uncorrelated_when_specific_variance_dominates():
    """Two assets with identical factor loadings but effectively zero shared factor variance and
    large, independent idiosyncratic variance should show up as ~uncorrelated — correlation must
    come from the shared covariance structure, not merely from matching betas."""
    loadings = np.array([
        [1.0],
        [1.0],
    ])
    factor_cov = np.array([[1e-8]])  # negligible shared factor variance
    specific_vars = np.array([0.05, 0.05])

    res = RiskDecompositionEngine.compute_correlation_matrix(loadings, factor_cov, specific_vars, symbols=["A", "B"])
    assert abs(res["correlation_matrix"][0][1]) < 0.01


def test_parametric_var_cvar_ordering():
    """CVaR (Expected Shortfall) must always be >= VaR at the same confidence/horizon, and both
    must grow with confidence level and with horizon (square-root-of-time scaling)."""
    daily_variance = (0.20 ** 2) / 252  # ~20% annualized vol

    var_95 = RiskDecompositionEngine.compute_parametric_var(daily_variance, confidence_level=0.95, horizon_days=1)
    var_99 = RiskDecompositionEngine.compute_parametric_var(daily_variance, confidence_level=0.99, horizon_days=1)
    var_95_10d = RiskDecompositionEngine.compute_parametric_var(daily_variance, confidence_level=0.95, horizon_days=10)

    assert var_95["cvar_pct"] >= var_95["var_pct"]
    assert var_99["var_pct"] > var_95["var_pct"]
    assert var_95_10d["var_pct"] > var_95["var_pct"]
    # sqrt(10) scaling should hold exactly for the parametric formula
    assert np.isclose(var_95_10d["var_pct"], var_95["var_pct"] * np.sqrt(10))