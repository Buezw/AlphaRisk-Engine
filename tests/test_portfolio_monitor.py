import numpy as np
import pandas as pd
import pytest

from src.db_client import RiskDatabase
from src.portfolio_monitor import PortfolioMonitor


def _build_db(tmp_path) -> RiskDatabase:
    db = RiskDatabase(str(tmp_path / "test.db"))
    db.upsert_securities(["AAA", "BBB"])

    factor_names = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    betas = {
        "AAA": [1.2, 0.1, -0.2, 0.1, 0.0, 0.3],
        "BBB": [0.8, -0.1, 0.3, 0.2, 0.1, -0.1],
    }
    exposure_rows, stats_rows = [], []
    for symbol, b in betas.items():
        for name, beta in zip(factor_names, b):
            exposure_rows.append({"symbol": symbol, "as_of_date": "2026-01-01", "factor_name": name, "beta": beta})
        stats_rows.append({"symbol": symbol, "as_of_date": "2026-01-01", "idiosyncratic_var": 0.0005, "r_squared": 0.5})
    db.upsert_exposures_and_stats(exposure_rows, stats_rows)

    rng = np.random.RandomState(0)
    n = 50
    factors_wide = pd.DataFrame({
        "trade_date": [f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}" for i in range(n)],
        "MKT": rng.normal(0, 0.01, n),
        "SMB": rng.normal(0, 0.005, n),
        "HML": rng.normal(0, 0.005, n),
        "RMW": rng.normal(0, 0.005, n),
        "CMA": rng.normal(0, 0.005, n),
        "MOM": rng.normal(0, 0.008, n),
    })
    db.upsert_factors_narrow(factors_wide)
    return db


def test_compute_portfolio_risk_returns_expected_shape(tmp_path):
    db = _build_db(tmp_path)
    risk = PortfolioMonitor.compute_portfolio_risk(db, {"AAA": 0.5, "BBB": 0.5})
    db.close()

    assert set(risk["symbols"]) == {"AAA", "BBB"}
    assert risk["abs_res"]["annualized_vol"] > 0
    assert set(risk["rc_res"]["risk_contribution_pct"]) == {"AAA", "BBB"}
    assert risk["var_1d_95"]["var_pct"] > 0
    assert risk["var_1d_99"]["var_pct"] > risk["var_1d_95"]["var_pct"]


def test_compute_portfolio_risk_missing_symbol_raises(tmp_path):
    db = _build_db(tmp_path)
    with pytest.raises(ValueError):
        PortfolioMonitor.compute_portfolio_risk(db, {"AAA": 0.5, "ZZZ": 0.5})
    db.close()


def test_compute_portfolio_risk_normalizes_weights(tmp_path):
    db = _build_db(tmp_path)
    risk_a = PortfolioMonitor.compute_portfolio_risk(db, {"AAA": 1, "BBB": 1})
    risk_b = PortfolioMonitor.compute_portfolio_risk(db, {"AAA": 3, "BBB": 3})
    db.close()
    assert risk_a["weights"] == pytest.approx(risk_b["weights"])


def test_check_budget_flags_breaches():
    risk = {
        "abs_res": {
            "annualized_vol": 0.25,
            "specific_var_ratio": 0.7,
            "portfolio_exposure": {"MKT": 1.6, "MOM": 0.2},
        },
        "rc_res": {"risk_contribution_pct": {"AAA": 0.5, "BBB": 0.5}},
        "var_1d_95": {"var_pct": 0.04},
    }
    budget = {
        "max_annualized_vol": 0.20,
        "max_var_1d_95": 0.03,
        "max_specific_var_ratio": 0.6,
        "max_single_stock_risk_pct": 0.4,
        "max_factor_exposure": {"MKT": 1.5},
    }
    checks = PortfolioMonitor.check_budget(risk, budget)
    breached = {c["check"] for c in checks if c["breached"]}

    assert "annualized_vol" in breached
    assert "var_1d_95" in breached
    assert "specific_var_ratio" in breached
    assert "single_stock_risk_pct[AAA]" in breached
    assert "single_stock_risk_pct[BBB]" in breached
    assert "factor_exposure[MKT]" in breached


def test_check_budget_no_breach_when_within_limits():
    risk = {
        "abs_res": {"annualized_vol": 0.10, "specific_var_ratio": 0.3, "portfolio_exposure": {"MKT": 1.0}},
        "rc_res": {"risk_contribution_pct": {"AAA": 0.5, "BBB": 0.5}},
        "var_1d_95": {"var_pct": 0.01},
    }
    budget = {"max_annualized_vol": 0.20, "max_var_1d_95": 0.03, "max_single_stock_risk_pct": 0.6}
    checks = PortfolioMonitor.check_budget(risk, budget)
    assert checks and all(not c["breached"] for c in checks)


def test_check_budget_unknown_factor_raises():
    risk = {
        "abs_res": {"annualized_vol": 0.1, "specific_var_ratio": 0.1, "portfolio_exposure": {"MKT": 1.0}},
        "rc_res": {"risk_contribution_pct": {}},
        "var_1d_95": {"var_pct": 0.01},
    }
    with pytest.raises(ValueError):
        PortfolioMonitor.check_budget(risk, {"max_factor_exposure": {"NOPE": 1.0}})
