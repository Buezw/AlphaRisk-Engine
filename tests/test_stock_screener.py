import pandas as pd
import pytest

from src.stock_screener import STYLE_PRESETS, StockScreener


def _make_universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "VAL1", "sector": "Financials", "HML": 0.5, "RMW": 0.1, "MKT": 1.0, "MOM": 0.0, "r_squared": 0.6, "idiosyncratic_var": 0.0004},
        {"symbol": "VAL2", "sector": "Financials", "HML": 0.4, "RMW": 0.05, "MKT": 0.9, "MOM": -0.1, "r_squared": 0.5, "idiosyncratic_var": 0.0003},
        {"symbol": "GRW1", "sector": "Technology", "HML": -0.3, "RMW": 0.3, "MKT": 1.3, "MOM": 0.4, "r_squared": 0.7, "idiosyncratic_var": 0.0006},
        {"symbol": "LOWQ", "sector": "Energy", "HML": 0.1, "RMW": -0.2, "MKT": 1.1, "MOM": 0.0, "r_squared": 0.1, "idiosyncratic_var": 0.001},
        {"symbol": "SPY", "sector": "Benchmark", "HML": 0.0, "RMW": 0.0, "MKT": 1.0, "MOM": 0.0, "r_squared": 1.0, "idiosyncratic_var": 0.0},
    ])


def test_resolve_style_single_preset():
    bounds = StockScreener.resolve_style(["value"])
    assert bounds == {"HML": (0.3, None)}


def test_resolve_style_unknown_raises():
    with pytest.raises(ValueError):
        StockScreener.resolve_style(["not_a_style"])


def test_merge_factor_bounds_tightens_overlapping_factor():
    merged = StockScreener.merge_factor_bounds(
        {"MKT": (None, 0.9)},
        {"MKT": (0.2, 1.5)},
    )
    # min side: tightest lower bound is 0.2 (only one specified so it wins); max side: 0.9 wins
    # over 1.5 since 0.9 is the tighter (smaller) ceiling.
    assert merged == {"MKT": (0.2, 0.9)}


def test_screen_filters_by_factor_bound_and_excludes_benchmark():
    df = _make_universe()
    result = StockScreener.screen(df, factor_bounds={"HML": (0.3, None)}, exclude_symbols=["SPY"], min_r_squared=0.0)
    assert set(result["symbol"]) == {"VAL1", "VAL2"}


def test_screen_min_r_squared_drops_low_fit_stocks():
    df = _make_universe()
    result = StockScreener.screen(df, min_r_squared=0.5, exclude_symbols=["SPY"])
    assert "LOWQ" not in set(result["symbol"])


def test_screen_max_idiosyncratic_var():
    df = _make_universe()
    result = StockScreener.screen(df, max_idiosyncratic_var=0.0005, exclude_symbols=["SPY"], min_r_squared=0.0)
    assert set(result["symbol"]) == {"VAL1", "VAL2"}


def test_screen_max_per_sector_caps_after_sort():
    df = _make_universe()
    result = StockScreener.screen(
        df, exclude_symbols=["SPY"], min_r_squared=0.0, sort_by="HML", ascending=False, max_per_sector=1
    )
    # Financials has VAL1 (0.5) and VAL2 (0.4); only the higher-HML one should survive the cap.
    financials = result[result["sector"] == "Financials"]
    assert list(financials["symbol"]) == ["VAL1"]


def test_screen_top_n_and_unknown_column_raises():
    df = _make_universe()
    result = StockScreener.screen(df, exclude_symbols=["SPY"], min_r_squared=0.0, sort_by="HML", top_n=1)
    assert len(result) == 1

    with pytest.raises(ValueError):
        StockScreener.screen(df, factor_bounds={"NOT_A_FACTOR": (0.1, None)})


def test_compute_composite_score_combines_zscored_factors():
    df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "HML": [0.0, 1.0, 2.0],
        "RMW": [2.0, 1.0, 0.0],
    })
    score = StockScreener.compute_composite_score(df, {"HML": 1.0, "RMW": 1.0})
    # HML z-scores: [-1, 0, 1] (std=1 for this evenly-spaced series); RMW z-scores: [1, 0, -1].
    # Summed with equal weight, every row's composite score should cancel to ~0.
    assert score.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_compute_composite_score_weights_favor_the_named_factor():
    df = pd.DataFrame({"symbol": ["A", "B", "C"], "HML": [0.0, 1.0, 2.0], "RMW": [0.0, 0.0, 0.0]})
    score = StockScreener.compute_composite_score(df, {"HML": 2.0, "RMW": 1.0})
    # RMW is constant (zero variance) so it contributes nothing; ranking should follow HML.
    assert score.tolist() == sorted(score.tolist())


def test_compute_composite_score_unknown_factor_raises():
    df = pd.DataFrame({"symbol": ["A"], "HML": [0.1]})
    with pytest.raises(ValueError):
        StockScreener.compute_composite_score(df, {"NOPE": 1.0})


def test_screen_composite_weights_overrides_sort_by_and_adds_score_column():
    df = _make_universe()
    result = StockScreener.screen(
        df, exclude_symbols=["SPY"], min_r_squared=0.0,
        composite_weights={"HML": 1.0, "RMW": 1.0}, sort_by="MOM",  # sort_by should be ignored
    )
    assert "composite_score" in result.columns
    # Ranked descending by composite_score by default.
    assert list(result["composite_score"]) == sorted(result["composite_score"], reverse=True)


def test_all_style_presets_reference_real_factor_names():
    known_factors = {"MKT", "SMB", "HML", "RMW", "CMA", "MOM"}
    for preset, bounds in STYLE_PRESETS.items():
        assert set(bounds) <= known_factors, f"preset '{preset}' references an unknown factor"
