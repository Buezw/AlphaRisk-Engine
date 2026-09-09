import numpy as np
import pandas as pd
import pytest

from src.strategy_backtest import MomentumBacktester


def _make_prices(n_weeks: int = 20, symbols=("A", "B", "C")) -> pd.DataFrame:
    """One trading day per week (Monday), so exec_dates == the whole index — keeps the math in
    each test easy to hand-check."""
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    data = {}
    # A always doubles each week (strongest momentum), B is flat, C always halves.
    data["A"] = 100.0 * (2.0 ** np.arange(n_weeks))
    data["B"] = np.full(n_weeks, 100.0)
    data["C"] = 100.0 * (0.5 ** np.arange(n_weeks))
    return pd.DataFrame(data, index=dates)[list(symbols)]


def test_build_weekly_execution_dates_picks_first_day_per_week():
    dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-10", "2024-01-17"])
    exec_dates = MomentumBacktester.build_weekly_execution_dates(dates)
    assert exec_dates == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-17")]


def test_run_selects_strongest_momentum_name():
    prices = _make_prices(n_weeks=10)
    result = MomentumBacktester.run(prices, top_k=1, lookback_weeks=2, require_full_history=False)
    # With top_k=1, every week should pick A (it always has the highest trailing return).
    for entry in result["selected_history"]:
        if entry["symbols"]:
            assert entry["symbols"] == ["A"]


def test_run_zero_cost_matches_gross_return():
    prices = _make_prices(n_weeks=10)
    result = MomentumBacktester.run(prices, top_k=1, lookback_weeks=2, cost_bps=0.0, require_full_history=False)
    weekly = result["weekly_returns"]
    assert (weekly["return"] == weekly["net_return"]).all()


def test_run_positive_cost_reduces_net_return_when_turnover_occurs():
    prices = _make_prices(n_weeks=10)
    result = MomentumBacktester.run(prices, top_k=1, lookback_weeks=2, cost_bps=50.0, require_full_history=False)
    weekly = result["weekly_returns"]
    turned_over = weekly[weekly["turnover"] > 0]
    assert not turned_over.empty
    assert (turned_over["net_return"] < turned_over["return"]).all()


def test_run_require_full_history_excludes_late_starting_symbol():
    prices = _make_prices(n_weeks=10, symbols=("A", "B", "C"))
    # D only starts trading halfway through — with require_full_history it must never be picked.
    prices["D"] = np.nan
    prices.loc[prices.index[5]:, "D"] = 1000.0 * (1.5 ** np.arange(len(prices) - 5))

    result = MomentumBacktester.run(prices, top_k=1, lookback_weeks=2, require_full_history=True)
    picked = {s for entry in result["selected_history"] for s in entry["symbols"]}
    assert "D" not in picked


def test_generate_current_picks_ranks_by_trailing_momentum():
    prices = _make_prices(n_weeks=10)
    result = MomentumBacktester.generate_current_picks(prices, top_k=2, lookback_weeks=3, require_full_history=False)
    picks = result["picks"]
    assert list(picks.sort_values("momentum_return", ascending=False).index)[0] == "A"
    assert abs(picks["target_weight"].sum() - 1.0) < 1e-9


def test_generate_current_picks_raises_on_empty_input():
    with pytest.raises(ValueError):
        MomentumBacktester.generate_current_picks(pd.DataFrame(), top_k=2, lookback_weeks=3)


def test_performance_stats_basic_identities():
    # Constant 1% weekly return for 52 weeks.
    r = pd.Series([0.01] * 52)
    stats = MomentumBacktester.performance_stats(r)
    assert stats["n_periods"] == 52
    assert stats["win_rate"] == 1.0
    assert stats["max_drawdown"] == 0.0
    assert stats["total_return"] == pytest.approx((1.01 ** 52) - 1, rel=1e-9)


def test_performance_stats_empty_series_returns_empty_dict():
    assert MomentumBacktester.performance_stats(pd.Series(dtype=float)) == {}
