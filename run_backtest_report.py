"""
End-to-end backtest diligence report — step 5 of the passive factor-allocation workflow.

Combines two independent checks against a chosen portfolio:
  1. Momentum-strategy backtest, split into an earlier development window and a later, untouched
     holdout window (see src/backtest_report.py) — flags a rule whose performance doesn't survive
     into a later period, rather than trusting one pooled backtest number.
  2. VaR backtest (Kupiec proportion-of-failures test) — checks whether the portfolio's realized
     daily losses over main.py's rolling ~1-year window actually breached the model's parametric
     VaR at the rate the model itself predicts.

Usage:
    python run_backtest_report.py --holdings holdings.json
    python run_backtest_report.py --holdings holdings.json --top-k 8 --lookback-weeks 8 --holdout-fraction 0.25
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.backtest_report import BacktestReport
from src.db_client import RiskDatabase


def _load_holdings(path: str) -> dict:
    try:
        with open(path) as f:
            holdings = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read holdings file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    if not holdings:
        print(f"Holdings file '{path}' is empty.", file=sys.stderr)
        sys.exit(1)
    return holdings


def _print_stats_row(label: str, stats: dict) -> None:
    if not stats:
        print(f"  {label}: not enough data")
        return
    print(
        f"  {label:<28} ann.return={stats['annualized_return']:+.1%}  "
        f"sharpe={stats['sharpe_ratio']:+.2f}  max_dd={stats['max_drawdown']:.1%}  "
        f"win_rate={stats['win_rate']:.0%}  n_weeks={stats['n_periods']}"
    )


def run_momentum_section(db: RiskDatabase, args) -> None:
    print("\n=== 1. Momentum Strategy: Development vs. Holdout ===")
    prices_wide = db.load_backtest_prices_wide()
    if prices_wide.empty:
        print("  No long-horizon history — run `python fetch_backtest_history.py --years 5` first. Skipped.")
        return
    # load_backtest_prices_wide() returns trade_date as plain strings (it's a raw SQL pivot) —
    # MomentumBacktester needs a real DatetimeIndex for isocalendar()-based weekly grouping and
    # for .loc lookups keyed by the Timestamps its own exec-date logic produces internally.
    prices_wide.index = pd.to_datetime(prices_wide.index)
    universe = prices_wide.drop(columns=["SPY"], errors="ignore")
    try:
        oos = BacktestReport.run_momentum_oos(
            universe, top_k=args.top_k, lookback_weeks=args.lookback_weeks,
            cost_bps=args.cost_bps, holdout_fraction=args.holdout_fraction,
        )
    except ValueError as e:
        print(f"  Could not run backtest: {e}")
        return

    print(f"  Split at {oos['cutoff_date'].date()} (holdout = most recent {args.holdout_fraction:.0%} of the sample).")
    _print_stats_row("Development (net of costs)", oos["development"]["net_stats"])
    _print_stats_row("Holdout (net of costs)", oos["holdout"]["net_stats"])

    dev_sharpe = oos["development"]["net_stats"].get("sharpe_ratio")
    hold_sharpe = oos["holdout"]["net_stats"].get("sharpe_ratio")
    if dev_sharpe is not None and hold_sharpe is not None and dev_sharpe > 0.5 and hold_sharpe < 0:
        print(
            "  ⚠️  Sharpe ratio flips from solidly positive in development to negative in "
            "the holdout window — treat the development-period result as regime-specific, not a durable edge."
        )


def run_var_section(db: RiskDatabase, holdings: dict, args) -> None:
    print("\n=== 2. VaR Backtest (Kupiec proportion-of-failures test) ===")
    symbols = list(holdings.keys())
    weights = np.array([holdings[s] for s in symbols], dtype=float)
    weights = weights / weights.sum()

    exposures_df = db.load_exposures_wide()
    factors_df = db.load_factors_wide()
    if exposures_df.empty:
        print("  No factor exposures in the database — run `python main.py` first. Skipped.")
        return
    missing = [s for s in symbols if s not in set(exposures_df["symbol"])]
    if missing:
        print(f"  Symbols missing factor exposures (skipping VaR check): {missing}")
        return

    port_exp = exposures_df.set_index("symbol").loc[symbols].reset_index()
    factor_cols = [c for c in port_exp.columns if c not in {"symbol", "as_of_date", "idiosyncratic_var", "r_squared"}]
    B_port = port_exp[factor_cols].values
    specific_vars = port_exp["idiosyncratic_var"].values

    factors_clean = factors_df[[c for c in factor_cols if c in factors_df.columns]] if not factors_df.empty else factors_df
    if len(factors_clean) > 10:
        factor_cov = np.cov(factors_clean.values, rowvar=False)
    else:
        factor_cov = np.eye(len(factor_cols)) * 0.0001

    from src.risk_decomposition import RiskDecompositionEngine
    abs_res = RiskDecompositionEngine.decompose_risk(weights, B_port, factor_cov, specific_vars, factor_cols)
    returns_wide = db.load_returns_wide(symbols)

    result = BacktestReport.run_var_check(returns_wide, symbols, weights, abs_res["total_variance"], args.confidence)

    if result["n_obs"] == 0:
        print("  Not enough overlapping daily-return history for these holdings to backtest yet.")
        return

    print(f"  Observed violation rate: {result['violation_rate']:.1%} (expected {result['expected_rate']:.1%})")
    print(f"  Violations / observations: {result['n_violations']} / {result['n_obs']}")
    print(
        f"  Kupiec test: {'REJECTED' if result['reject_null'] else 'PASSED'} "
        f"(p={result['p_value']:.4f}), traffic light: {result['traffic_light']}"
    )
    if result["reject_null"]:
        print(
            "  ⚠️  The normal-distribution VaR model is statistically miscalibrated for "
            "this portfolio's realized returns — treat the VaR/CVaR figures in app.py as "
            "understating tail risk, not as a hard ceiling."
        )


def main():
    parser = argparse.ArgumentParser(description="Combined momentum-strategy + VaR backtest diligence report.")
    parser.add_argument("--holdings", default="holdings.json", help="JSON file of {symbol: weight}.")
    parser.add_argument("--db", default="factor_risk.db")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    holdings = _load_holdings(args.holdings)

    try:
        with RiskDatabase(args.db) as db:
            run_momentum_section(db, args)
            run_var_section(db, holdings, args)
    except Exception as e:
        print(f"Report failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
