import numpy as np
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskDecompositionEngine:
    """Computes portfolio total variance, active risk (Tracking Error), and factor attributions."""

    @staticmethod
    def decompose_risk(weights: np.ndarray, loadings: np.ndarray, factor_cov: np.ndarray, specific_vars: np.ndarray, factor_names: list[str] = None) -> dict:
        """Computes the absolute-risk variance decomposition."""
        w = np.asarray(weights)
        B = np.asarray(loadings)
        Sigma_F = np.asarray(factor_cov)
        D = np.diag(specific_vars)

        factor_var = float(w.T @ (B @ Sigma_F @ B.T) @ w)
        specific_var = float(w.T @ D @ w)
        total_var = factor_var + specific_var

        # Portfolio-level factor exposure (B_p = B^T * w)
        # A weighted linear combination of each stock's factor exposure — the six-factor profile of
        # "the portfolio as a whole"
        portfolio_exposure = B.T @ w

        result = {
            "total_variance": total_var,
            "annualized_vol": float(np.sqrt(total_var * 252)),
            "factor_variance": factor_var,
            "specific_variance": specific_var,
            "factor_var_ratio": factor_var / total_var if total_var > 0 else 0.0,
            "specific_var_ratio": specific_var / total_var if total_var > 0 else 0.0,
        }

        if factor_names is not None:
            result["portfolio_exposure"] = dict(zip(factor_names, portfolio_exposure.tolist()))
        else:
            result["portfolio_exposure"] = portfolio_exposure.tolist()

        return result

    @staticmethod
    def compute_risk_contributions(
        weights: np.ndarray,
        loadings: np.ndarray,
        factor_cov: np.ndarray,
        specific_vars: np.ndarray,
        symbols: list[str] = None
    ) -> dict:
        """
        Euler decomposition of portfolio volatility into per-holding marginal contributions.

        Reconstructs the full N x N asset covariance matrix Σ = B Σ_F Bᵀ + D, then uses the fact
        that portfolio volatility σ_p = sqrt(wᵀ Σ w) is homogeneous of degree 1 in w, so it can be
        split additively across holdings:
            MCTR_i = (Σw)_i / σ_p               (marginal contribution to risk: ∂σ_p / ∂w_i)
            RC_i   = w_i * MCTR_i                (risk contribution, in vol units — Σ RC_i = σ_p)
            RC_i / σ_p                            (risk contribution as a % of total portfolio vol)
        This tells you which holdings are actually driving portfolio risk, as opposed to just
        showing their weight or their standalone factor loadings.
        """
        w = np.asarray(weights)
        B = np.asarray(loadings)
        Sigma_F = np.asarray(factor_cov)
        D = np.diag(specific_vars)

        Sigma = B @ Sigma_F @ B.T + D  # full N x N daily asset covariance matrix
        port_var = float(w.T @ Sigma @ w)
        port_vol_daily = np.sqrt(port_var) if port_var > 0 else 0.0
        annualize = np.sqrt(252)

        if port_vol_daily > 0:
            mctr = (Sigma @ w) / port_vol_daily * annualize  # annualized marginal contribution to vol
        else:
            mctr = np.zeros_like(w)

        risk_contribution = w * mctr  # additive: sum(risk_contribution) == total_annualized_vol
        total_annualized_vol = port_vol_daily * annualize
        risk_contribution_pct = (
            risk_contribution / total_annualized_vol if total_annualized_vol > 0 else np.zeros_like(w)
        )

        result = {
            "total_annualized_vol": total_annualized_vol,
        }

        if symbols is not None:
            result["mctr"] = dict(zip(symbols, mctr.tolist()))
            result["risk_contribution"] = dict(zip(symbols, risk_contribution.tolist()))
            result["risk_contribution_pct"] = dict(zip(symbols, risk_contribution_pct.tolist()))
        else:
            result["mctr"] = mctr.tolist()
            result["risk_contribution"] = risk_contribution.tolist()
            result["risk_contribution_pct"] = risk_contribution_pct.tolist()

        return result

    @staticmethod
    def compute_correlation_matrix(
        loadings: np.ndarray,
        factor_cov: np.ndarray,
        specific_vars: np.ndarray,
        symbols: list[str] = None
    ) -> dict:
        """
        Pairwise return-correlation matrix implied by the full asset covariance matrix
        Σ = B Σ_F Bᵀ + D — the same Σ used by compute_risk_contributions(), just normalized to
        [-1, 1] instead of left in variance units:
            Corr_ij = Σ_ij / (σ_i * σ_j)
        Two stocks can share an identical factor loading and still show up here as weakly
        correlated, if their idiosyncratic (specific) variance dominates — this is the pairwise
        counterpart to why compute_risk_contributions() needed the full covariance matrix rather
        than isolated betas.
        """
        B = np.asarray(loadings)
        Sigma_F = np.asarray(factor_cov)
        D = np.diag(specific_vars)

        Sigma = B @ Sigma_F @ B.T + D
        std = np.sqrt(np.diag(Sigma))
        outer_std = np.outer(std, std)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(outer_std > 0, Sigma / outer_std, 0.0)
        np.fill_diagonal(corr, 1.0)  # exact 1.0 on the diagonal, avoids float noise from self-division

        result = {"correlation_matrix": corr.tolist()}
        result["symbols"] = list(symbols) if symbols is not None else list(range(len(std)))
        return result

    @staticmethod
    def compute_parametric_var(
        daily_variance: float,
        confidence_level: float = 0.95,
        horizon_days: int = 1
    ) -> dict:
        """
        Parametric (Gaussian) Value at Risk and Conditional VaR / Expected Shortfall.

        Assumes daily portfolio returns are normally distributed with zero mean (the standard
        short-horizon simplification — at daily/weekly horizons, drift is negligible next to
        volatility) and scales to the requested horizon via the square-root-of-time rule:
            horizon_vol = sqrt(daily_variance) * sqrt(horizon_days)
            VaR_pct  = z * horizon_vol                          (z = Φ⁻¹(confidence_level))
            CVaR_pct = horizon_vol * φ(z) / (1 - confidence_level)   (Expected Shortfall, the
                                                                       mean loss in the tail beyond VaR)
        Both are returned as positive fractions of portfolio value (a loss magnitude, not a
        signed return) and reuse the same daily variance already produced by decompose_risk().
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        daily_vol = np.sqrt(daily_variance) if daily_variance > 0 else 0.0
        horizon_vol = daily_vol * np.sqrt(horizon_days)

        z = norm.ppf(confidence_level)
        var_pct = float(z * horizon_vol)
        cvar_pct = float(horizon_vol * norm.pdf(z) / (1 - confidence_level))

        return {
            "confidence_level": confidence_level,
            "horizon_days": horizon_days,
            "horizon_vol": float(horizon_vol),
            "var_pct": var_pct,
            "cvar_pct": cvar_pct,
        }

    @staticmethod
    def decompose_active_risk(
        w_portfolio: np.ndarray,
        w_benchmark: np.ndarray,
        loadings: np.ndarray,
        factor_cov: np.ndarray,
        specific_vars: np.ndarray,
        factor_names: list[str]
    ) -> dict:
        """
        Computes the active risk / tracking-error decomposition relative to a benchmark.
        """
        w_p = np.asarray(w_portfolio)
        w_b = np.asarray(w_benchmark)
        delta_w = w_p - w_b  # active weight vector, Δw

        B = np.asarray(loadings)
        Sigma_F = np.asarray(factor_cov)
        D = np.diag(specific_vars)

        # 1. Quadratic-form decomposition of the tracking error
        active_factor_var = float(delta_w.T @ (B @ Sigma_F @ B.T) @ delta_w)
        active_specific_var = float(delta_w.T @ D @ delta_w)
        active_total_var = active_factor_var + active_specific_var

        # 2. Annualized tracking error
        tracking_error = float(np.sqrt(active_total_var * 252))

        # 3. Net factor-exposure deviation (Active Factor Tilt: Δβ = B^T * Δw)
        active_tilts = B.T @ delta_w

        return {
            "tracking_error": tracking_error,
            "active_total_var": active_total_var,
            "active_factor_var": active_factor_var,
            "active_specific_var": active_specific_var,
            "active_factor_ratio": active_factor_var / active_total_var if active_total_var > 0 else 0.0,
            "active_specific_ratio": active_specific_var / active_total_var if active_total_var > 0 else 0.0,
            "active_tilts": dict(zip(factor_names, active_tilts.tolist())),
            "delta_w": delta_w.tolist()
        }