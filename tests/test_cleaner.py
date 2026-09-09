import numpy as np
import pandas as pd

from src.cleaner import FinancialDataCleaner


def test_clean_raw_quotes_drops_non_positive_prices():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    raw = pd.DataFrame({
        "symbol": ["A"] * 5,
        "trade_date": dates,
        "adj_close": [100.0, -5.0, 0.0, 101.0, 102.0],
    })
    cleaned = FinancialDataCleaner.clean_raw_quotes(raw)
    assert (cleaned["adj_close"] > 0).all()


def test_clean_raw_quotes_computes_daily_return():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    raw = pd.DataFrame({
        "symbol": ["A"] * 3,
        "trade_date": dates,
        "adj_close": [100.0, 110.0, 99.0],
    })
    cleaned = FinancialDataCleaner.clean_raw_quotes(raw)
    np.testing.assert_allclose(cleaned["daily_return"].values, [0.10, -0.10], atol=1e-9)


def test_clean_raw_quotes_clips_outliers_per_symbol_not_globally():
    """Regression test: outlier clipping must be computed per-symbol. A pooled, cross-symbol
    quantile would let one stock's corrupt spike distort the clipping bounds applied to a
    completely unrelated stock's legitimate returns."""
    dates = pd.date_range("2025-01-01", periods=31, freq="B")

    # STABLE: flat, except one unadjusted-split-like +200% spike partway through.
    stable_prices = [100.0] * 16 + [300.0] * 15
    # VOLATILE: a real, sizeable but legitimate ~+/-25% daily swinger — no single day over the 50% threshold.
    rng = np.random.RandomState(3)
    vol_returns = rng.uniform(-0.25, 0.25, 30)
    vol_prices = [100.0]
    for r in vol_returns:
        vol_prices.append(vol_prices[-1] * (1 + r))

    raw = pd.concat([
        pd.DataFrame({"symbol": "STABLE", "trade_date": dates, "adj_close": stable_prices}),
        pd.DataFrame({"symbol": "VOLATILE", "trade_date": dates, "adj_close": vol_prices}),
    ], ignore_index=True)

    cleaned = FinancialDataCleaner.clean_raw_quotes(raw)
    stable = cleaned[cleaned["symbol"] == "STABLE"]
    volatile = cleaned[cleaned["symbol"] == "VOLATILE"]

    # STABLE's corrupt +200% (2.0) spike must be clipped down toward its own distribution's 99th
    # percentile — with 31 mostly-zero observations, that lands around 1.4, well below the raw 2.0.
    assert stable["daily_return"].max() < 1.5

    # VOLATILE's own legitimate swings must survive uncapped by STABLE's outlier — proof the
    # clip bound came from VOLATILE's own distribution, not a pooled cross-symbol quantile.
    assert volatile["daily_return"].abs().max() > 0.20
