import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Computes portfolio weights directly from the risk model's covariance matrix
    Σ = B Σ_F Bᵀ + D — no expected-return forecast involved anywhere, consistent with this
    project's stance of not predicting returns. Both methods below only need the same full
    covariance matrix already used everywhere else in the risk engine (decompose_risk,
    compute_risk_contributions, etc.); they answer "how should I weight what I've already
    decided to hold," not "what should I hold."
    """

    @staticmethod
    def build_covariance(loadings: np.ndarray, factor_cov: np.ndarray, specific_vars: np.ndarray) -> np.ndarray:
        B = np.asarray(loadings)
        Sigma_F = np.asarray(factor_cov)
        D = np.diag(specific_vars)
        return B @ Sigma_F @ B.T + D

    @staticmethod
    def _check_feasible(max_weight: float, n: int) -> None:
        if max_weight * n < 1.0 - 1e-9:
            raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets (need max_weight >= 1/{n} = {1.0 / n:.4f}).")

    @staticmethod
    def min_variance_weights(
        loadings: np.ndarray, factor_cov: np.ndarray, specific_vars: np.ndarray, max_weight: float = 1.0
    ) -> np.ndarray:
        """Long-only global minimum-variance portfolio: min w'Σw s.t. sum(w)=1, 0<=w<=max_weight.

        Solved numerically (SLSQP) rather than the textbook closed form w = Σ⁻¹1 / (1ᵀΣ⁻¹1),
        because the closed form ignores the long-only / max_weight box constraints entirely and
        can return negative or arbitrarily concentrated weights that violate them.
        """
        Sigma = PortfolioOptimizer.build_covariance(loadings, factor_cov, specific_vars)
        n = Sigma.shape[0]
        PortfolioOptimizer._check_feasible(max_weight, n)

        w0 = np.full(n, 1.0 / n)
        result = minimize(
            lambda w: float(w @ Sigma @ w),
            w0,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 500, "ftol": 1e-14},
        )
        if not result.success:
            raise RuntimeError(f"Minimum-variance optimization failed to converge: {result.message}")
        w = np.clip(result.x, 0.0, max_weight)
        return w / w.sum()

    @staticmethod
    def risk_parity_weights(
        loadings: np.ndarray, factor_cov: np.ndarray, specific_vars: np.ndarray, max_weight: float = 1.0
    ) -> np.ndarray:
        """Long-only equal-risk-contribution portfolio: finds w (sum=1, 0<=w<=max_weight) that
        minimizes the spread between every holding's Euler risk contribution RC_i = w_i*(Σw)_i
        and the equal-split target (total portfolio variance / n). Unlike min-variance, this
        doesn't collapse weight onto the single lowest-vol asset — every holding ends up
        contributing roughly the same share of total risk, which is the point of the method.
        No expected-return input, same as min-variance.
        """
        Sigma = PortfolioOptimizer.build_covariance(loadings, factor_cov, specific_vars)
        n = Sigma.shape[0]
        PortfolioOptimizer._check_feasible(max_weight, n)

        def _objective(w):
            port_var = w @ Sigma @ w
            rc = w * (Sigma @ w)
            target = port_var / n
            return float(np.sum((rc - target) ** 2))

        w0 = np.full(n, 1.0 / n)
        result = minimize(
            _objective, w0, method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-18},
        )
        if not result.success:
            raise RuntimeError(f"Risk-parity optimization failed to converge: {result.message}")
        w = np.clip(result.x, 0.0, max_weight)
        return w / w.sum()
