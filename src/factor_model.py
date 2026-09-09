import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FactorExposureEngine:
    """
    Computes multi-factor exposures using pure Linear Algebra (Normal Equations).
    Includes numerical safeguards like Ridge (Tikhonov) regularization for ill-conditioned matrices.
    """
    
    @staticmethod
    def solve_ols(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-4) -> Tuple[np.ndarray, float, float]:
        """
        Solves OLS via Normal Equation: beta = (X^T * X)^(-1) * X^T * y
        Falls back to Ridge Regression if high multi-collinearity is detected.
        
        Args:
            X: Matrix of shape (N, K) containing factor returns (including intercept).
            y: Vector of shape (N,) containing asset excess returns.
            ridge_lambda: Shrinkage parameter for L2 regularization.
            
        Returns:
            Tuple of (beta_coefficients, idiosyncratic_variance, r_squared)
        """
        N, K = X.shape
        if N <= K:
            raise ValueError(f"Degrees of freedom error: N ({N}) must be greater than K ({K}).")

        XtX = np.dot(X.T, X)
        
        # Numerical-stability safeguard: check the matrix condition number
        # The higher the condition number, the closer the matrix is to singular, and the more the
        # floating-point error in the inverse blows up exponentially
        cond_num = np.linalg.cond(XtX)
        if cond_num > 1e6:
            logger.warning(
                f"Ill-conditioned matrix detected (Condition Number: {cond_num:.1e}). "
                f"Applying Ridge Regularization (lambda={ridge_lambda})."
            )
            # Trigger Tikhonov (ridge) regularization: add a small perturbation lambda to the main diagonal
            XtX += ridge_lambda * np.eye(K)

        # Safe inverse
        XtX_inv = np.linalg.inv(XtX)
        betas = np.dot(np.dot(XtX_inv, X.T), y)

        # Compute residuals
        residuals = y - np.dot(X, betas)

        # Idiosyncratic variance (unbiased estimate, N - K degrees of freedom)
        idiosyncratic_var = float(np.dot(residuals.T, residuals) / (N - K))

        # Goodness of fit, R^2
        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(residuals ** 2))
        r_squared = 1.0 - (ss_res / ss_total) if ss_total > 0 else 0.0
        
        return betas, idiosyncratic_var, r_squared