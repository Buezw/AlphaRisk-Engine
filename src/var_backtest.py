import numpy as np
import pandas as pd
from scipy.stats import chi2


class VarBacktester:
    """
    Backtests a parametric VaR estimate against realized portfolio returns.

    RiskDecompositionEngine.compute_parametric_var() assumes daily returns are normally
    distributed. Equity returns are fat-tailed and negatively skewed, so that assumption can
    understate tail risk — the only way to know whether it actually does, for this portfolio,
    is to check how often realized losses over the model's own lookback window actually
    exceeded the VaR it would have predicted. This module runs that check (a proportion-of-
    failures test) rather than trusting the normal-distribution assumption on faith.
    """

    @staticmethod
    def reconstruct_portfolio_returns(returns_wide: pd.DataFrame, symbols: list[str], weights: np.ndarray) -> pd.Series:
        """
        Weighted sum of each holding's historical daily return, restricted to dates where every
        holding has a return (an incomplete row would understate the portfolio's actual daily
        move on that date, biasing the violation count).
        """
        missing = [s for s in symbols if s not in returns_wide.columns]
        if missing or returns_wide.empty:
            return pd.Series(dtype=float)
        aligned = returns_wide[symbols].dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        return pd.Series(aligned.values @ np.asarray(weights), index=aligned.index)

    @staticmethod
    def count_violations(portfolio_returns: pd.Series, var_pct: float) -> pd.Series:
        """A violation is a day where the realized loss exceeded the predicted VaR (var_pct is a positive loss fraction)."""
        return (-portfolio_returns) > var_pct

    @staticmethod
    def kupiec_pof_test(n_obs: int, n_violations: int, confidence_level: float) -> dict:
        """
        Kupiec (1995) unconditional-coverage / proportion-of-failures likelihood-ratio test.

        H0: the true exceedance probability equals the model's expected rate p = 1 - confidence_level.
            LR_pof = -2 * ln[ (1-p)^(n-x) * p^x / (1-x/n)^(n-x) * (x/n)^x ]  ~ chi2(1) under H0
        A small p-value (< 0.05) means the observed violation rate is statistically inconsistent
        with what the model claims — i.e. the VaR model is miscalibrated, not just "unlucky."
        """
        if n_obs == 0:
            return {
                "n_obs": 0, "n_violations": 0, "violation_rate": None,
                "expected_rate": 1 - confidence_level, "lr_stat": None,
                "p_value": None, "reject_null": None,
            }

        p = 1 - confidence_level
        x = n_violations
        n = n_obs
        pi_hat = x / n

        def _log_lik(pi: float) -> float:
            pi = min(max(pi, 1e-10), 1 - 1e-10)
            return (n - x) * np.log(1 - pi) + x * np.log(pi)

        lr_stat = -2 * (_log_lik(p) - _log_lik(pi_hat))
        p_value = float(1 - chi2.cdf(lr_stat, df=1))

        return {
            "n_obs": n,
            "n_violations": x,
            "violation_rate": pi_hat,
            "expected_rate": p,
            "lr_stat": float(lr_stat),
            "p_value": p_value,
            "reject_null": bool(p_value < 0.05),
        }

    @staticmethod
    def traffic_light(n_obs: int, n_violations: int, confidence_level: float) -> str:
        """
        Basel-style traffic-light zone, generalized from the fixed 250-day/99% table (0-4 green,
        5-9 yellow, 10+ red) to an arbitrary sample size and confidence level by comparing the
        observed violation count to a multiple of the model's own expected count, rather than a
        hardcoded threshold that only means something at exactly 250 observations and 99%.
        """
        expected = n_obs * (1 - confidence_level)
        if n_obs == 0:
            return "insufficient_data"
        if expected <= 0:
            expected = 1e-9
        ratio = n_violations / expected
        if ratio <= 1.5:
            return "green"
        elif ratio <= 3.0:
            return "yellow"
        else:
            return "red"

    @staticmethod
    def run(portfolio_returns: pd.Series, var_pct: float, confidence_level: float) -> dict:
        """Runs the full backtest: violation flags + Kupiec test + traffic-light verdict."""
        returns = portfolio_returns.dropna()
        violations = VarBacktester.count_violations(returns, var_pct)
        n_obs = len(returns)
        n_violations = int(violations.sum())

        result = VarBacktester.kupiec_pof_test(n_obs, n_violations, confidence_level)
        result["traffic_light"] = VarBacktester.traffic_light(n_obs, n_violations, confidence_level)
        result["returns"] = returns
        result["violations"] = violations
        return result
