import logging

import numpy as np
import pandas as pd

from src.strategy_backtest import MomentumBacktester
from src.var_backtest import VarBacktester
from src.risk_decomposition import RiskDecompositionEngine

logger = logging.getLogger(__name__)


class BacktestReport:
    """Ties the momentum-strategy backtest and the VaR backtest together into a single
    diligence report — step 5 of the README's passive factor-allocation workflow.

    The discipline this adds beyond calling MomentumBacktester.run() directly: it never reports
    a single pooled-period statistic as if it were proof a rule works. It splits the available
    history into an earlier "development" window and a later, untouched "holdout" window and
    reports both — a rule whose Sharpe / drawdown look very different between the two halves is
    a warning sign the pooled number was regime-specific luck, even though this simple momentum
    rule has no parameters *fit* to the data the way a machine-learning model would.
    """

    @staticmethod
    def run_momentum_oos(
        prices_wide: pd.DataFrame,
        top_k: int = 10,
        lookback_weeks: int = 12,
        cost_bps: float = 20.0,
        holdout_fraction: float = 0.3,
        require_full_history: bool = True,
    ) -> dict:
        """Runs MomentumBacktester.run() once over the full available history (so the lookback
        window right at the split boundary still sees legitimate trailing data from before the
        cutoff, avoiding an artificial discontinuity), then splits the resulting weekly-return
        series by date into a development slice and a later, untouched holdout slice.

        Args mirror MomentumBacktester.run(), plus:
            holdout_fraction: the most recent fraction of the sample's calendar span (not row
                count) held out as the untouched validation window.

        Returns {'cutoff_date', 'development': {...}, 'holdout': {...}, 'full_period': {...}},
        each of the three carrying 'weekly_returns', 'gross_stats', 'net_stats'.
        """
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be strictly between 0 and 1")

        bt_result = MomentumBacktester.run(
            prices_wide, top_k=top_k, lookback_weeks=lookback_weeks,
            cost_bps=cost_bps, require_full_history=require_full_history,
        )
        weekly = bt_result["weekly_returns"]
        if weekly.empty:
            raise ValueError("Backtest produced no weekly returns to split.")

        span = weekly.index.max() - weekly.index.min()
        cutoff = weekly.index.max() - span * holdout_fraction
        development = weekly[weekly.index <= cutoff]
        holdout = weekly[weekly.index > cutoff]

        if development.empty or holdout.empty:
            raise ValueError(
                f"Not enough weekly history ({len(weekly)} weeks) to split at "
                f"holdout_fraction={holdout_fraction} into two non-empty windows."
            )

        def _bundle(slice_df: pd.DataFrame) -> dict:
            return {
                "weekly_returns": slice_df,
                "gross_stats": MomentumBacktester.performance_stats(slice_df["return"]),
                "net_stats": MomentumBacktester.performance_stats(slice_df["net_return"]),
            }

        return {
            "cutoff_date": cutoff,
            "development": _bundle(development),
            "holdout": _bundle(holdout),
            "full_period": _bundle(weekly),
        }

    @staticmethod
    def run_var_check(
        returns_wide: pd.DataFrame,
        symbols: list,
        weights: np.ndarray,
        daily_variance: float,
        confidence_level: float = 0.95,
    ) -> dict:
        """Reconstructs the portfolio's historical daily returns and backtests the parametric
        1-day VaR implied by daily_variance (as produced by RiskDecompositionEngine.decompose_risk)
        against them via the Kupiec proportion-of-failures test."""
        var_res = RiskDecompositionEngine.compute_parametric_var(
            daily_variance, confidence_level=confidence_level, horizon_days=1
        )
        port_returns = VarBacktester.reconstruct_portfolio_returns(returns_wide, symbols, weights)
        return VarBacktester.run(port_returns, var_res["var_pct"], confidence_level)
