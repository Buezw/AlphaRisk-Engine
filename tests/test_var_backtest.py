import numpy as np
import pandas as pd
from src.var_backtest import VarBacktester


def test_kupiec_accepts_well_calibrated_model():
    """If the observed violation rate matches the model's expected rate exactly, Kupiec should not reject."""
    n_obs = 1000
    confidence_level = 0.95
    n_violations = int(n_obs * (1 - confidence_level))  # exactly 50 violations, matching the model's own claim

    result = VarBacktester.kupiec_pof_test(n_obs, n_violations, confidence_level)

    assert result["p_value"] > 0.05
    assert result["reject_null"] is False


def test_kupiec_rejects_severely_miscalibrated_model():
    """A model claiming 5% exceedance that actually breaches 20% of the time should be rejected."""
    n_obs = 500
    confidence_level = 0.95
    n_violations = 100  # 20% observed vs. 5% expected

    result = VarBacktester.kupiec_pof_test(n_obs, n_violations, confidence_level)

    assert result["p_value"] < 0.05
    assert result["reject_null"] is True


def test_kupiec_handles_zero_observations():
    result = VarBacktester.kupiec_pof_test(0, 0, 0.95)
    assert result["n_obs"] == 0
    assert result["p_value"] is None


def test_count_violations_flags_losses_beyond_var():
    returns = pd.Series([0.01, -0.02, -0.10, 0.005, -0.03])
    var_pct = 0.05  # a 5% predicted loss threshold

    violations = VarBacktester.count_violations(returns, var_pct)

    # only the -0.10 day breaches a 5% VaR
    assert violations.tolist() == [False, False, True, False, False]


def test_reconstruct_portfolio_returns_weighted_sum():
    returns_wide = pd.DataFrame(
        {"AAPL": [0.01, 0.02], "MSFT": [-0.01, 0.03]},
        index=["2024-01-01", "2024-01-02"],
    )
    weights = np.array([0.5, 0.5])

    port_returns = VarBacktester.reconstruct_portfolio_returns(returns_wide, ["AAPL", "MSFT"], weights)

    np.testing.assert_allclose(port_returns.values, [0.0, 0.025])


def test_reconstruct_portfolio_returns_missing_symbol_returns_empty():
    returns_wide = pd.DataFrame({"AAPL": [0.01]}, index=["2024-01-01"])
    result = VarBacktester.reconstruct_portfolio_returns(returns_wide, ["AAPL", "TSLA"], np.array([0.5, 0.5]))
    assert result.empty


def test_traffic_light_zones():
    confidence_level = 0.99
    n_obs = 250
    expected = n_obs * (1 - confidence_level)  # 2.5

    assert VarBacktester.traffic_light(n_obs, int(expected), confidence_level) == "green"
    assert VarBacktester.traffic_light(n_obs, int(expected * 2), confidence_level) == "yellow"
    assert VarBacktester.traffic_light(n_obs, int(expected * 5), confidence_level) == "red"


def test_run_end_to_end():
    np.random.seed(7)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var_pct = 0.0164485  # ~95% VaR for sigma=0.01 (z=1.645)

    result = VarBacktester.run(returns, var_pct, confidence_level=0.95)

    assert result["n_obs"] == 1000
    assert 0.0 <= result["violation_rate"] <= 0.15
    assert result["traffic_light"] in {"green", "yellow", "red"}
