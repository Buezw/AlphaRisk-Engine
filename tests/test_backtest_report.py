import numpy as np
import pandas as pd
import pytest

from src.backtest_report import BacktestReport


def _make_prices(n_weeks: int = 30, symbols=("A", "B", "C")) -> pd.DataFrame:
    """Same construction as test_strategy_backtest.py's helper: one trading day per week, A
    always doubling (strongest momentum), B flat, C always halving."""
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    data = {
        "A": 100.0 * (2.0 ** np.arange(n_weeks)),
        "B": np.full(n_weeks, 100.0),
        "C": 100.0 * (0.5 ** np.arange(n_weeks)),
    }
    return pd.DataFrame(data, index=dates)[list(symbols)]


def test_run_momentum_oos_invalid_fraction_raises():
    prices = _make_prices(n_weeks=30)
    for bad_fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            BacktestReport.run_momentum_oos(prices, top_k=1, lookback_weeks=2, holdout_fraction=bad_fraction)


def test_run_momentum_oos_splits_by_date_into_two_nonempty_windows():
    prices = _make_prices(n_weeks=30)
    result = BacktestReport.run_momentum_oos(
        prices, top_k=1, lookback_weeks=4, cost_bps=10.0, holdout_fraction=0.3, require_full_history=False
    )

    dev = result["development"]["weekly_returns"]
    hold = result["holdout"]["weekly_returns"]
    cutoff = result["cutoff_date"]

    assert not dev.empty
    assert not hold.empty
    assert (dev.index <= cutoff).all()
    assert (hold.index > cutoff).all()
    # full_period must recombine to exactly the same rows as development + holdout.
    assert len(dev) + len(hold) == len(result["full_period"]["weekly_returns"])

    assert result["development"]["net_stats"]
    assert result["holdout"]["net_stats"]


def test_run_momentum_oos_single_row_result_raises_on_split():
    # lookback_weeks + 2 weeks of price history produces exactly one weekly-return row (the
    # minimum MomentumBacktester.run will accept) — one row can never satisfy both halves of a
    # non-trivial split, so this must raise rather than silently return an empty holdout.
    prices = _make_prices(n_weeks=4)
    with pytest.raises(ValueError):
        BacktestReport.run_momentum_oos(prices, top_k=1, lookback_weeks=2, holdout_fraction=0.3, require_full_history=False)


def test_run_var_check_returns_kupiec_result():
    np.random.seed(11)
    n = 500
    returns_wide = pd.DataFrame({
        "AAPL": np.random.normal(0, 0.01, n),
        "MSFT": np.random.normal(0, 0.01, n),
    })
    weights = np.array([0.5, 0.5])
    daily_variance = 0.5 * (0.01 ** 2) + 0.5 * (0.01 ** 2)  # matches the equal-weighted, uncorrelated construction

    result = BacktestReport.run_var_check(returns_wide, ["AAPL", "MSFT"], weights, daily_variance, confidence_level=0.95)

    assert result["n_obs"] == n
    assert 0.0 <= result["violation_rate"] <= 0.2
    assert result["traffic_light"] in {"green", "yellow", "red"}
    assert "p_value" in result
