import logging

import numpy as np

from src.db_client import RiskDatabase
from src.risk_decomposition import RiskDecompositionEngine

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    """Recomputes a portfolio's current risk profile from factor_risk.db and checks it against a
    user-defined risk budget — the mechanical version of step 7 (periodic monitoring) in the
    README's passive factor-allocation workflow. Rerun this after every `python main.py` refresh
    instead of relying on remembering to reopen app.py and eyeball the numbers.
    """

    @staticmethod
    def compute_portfolio_risk(db: RiskDatabase, holdings: dict) -> dict:
        """
        Args:
            holdings: {symbol: weight}; weights need not already sum to 1 (renormalized here).

        Returns {'symbols', 'weights', 'abs_res', 'rc_res', 'var_1d_95', 'var_1d_99'} — the same
        RiskDecompositionEngine outputs app.py's Absolute Risk page is built from, computed fresh
        from whatever factor exposures and factor covariance are currently in the database.

        Raises ValueError if the database has no factor exposures yet, or if any holding has none.
        """
        symbols = list(holdings.keys())
        weights = np.array([holdings[s] for s in symbols], dtype=float)
        if weights.sum() <= 0:
            raise ValueError("Holdings weights must sum to a positive number.")
        weights = weights / weights.sum()

        exposures_df = db.load_exposures_wide()
        if exposures_df.empty:
            raise ValueError("No factor exposures in the database — run `python main.py` first.")

        missing = [s for s in symbols if s not in set(exposures_df["symbol"])]
        if missing:
            raise ValueError(f"These holdings have no factor exposures in the database: {missing}")

        port_exp = exposures_df.set_index("symbol").loc[symbols].reset_index()
        meta_cols = {"symbol", "as_of_date", "idiosyncratic_var", "r_squared"}
        factor_cols = [c for c in port_exp.columns if c not in meta_cols]
        B_port = port_exp[factor_cols].values
        specific_vars = port_exp["idiosyncratic_var"].values

        factors_df = db.load_factors_wide()
        if not factors_df.empty:
            factors_clean = factors_df[[c for c in factor_cols if c in factors_df.columns]]
        else:
            factors_clean = factors_df
        if len(factors_clean) > 10:
            factor_cov = np.cov(factors_clean.values, rowvar=False)
        else:
            factor_cov = np.eye(len(factor_cols)) * 0.0001

        abs_res = RiskDecompositionEngine.decompose_risk(weights, B_port, factor_cov, specific_vars, factor_cols)
        rc_res = RiskDecompositionEngine.compute_risk_contributions(weights, B_port, factor_cov, specific_vars, symbols)
        var_95 = RiskDecompositionEngine.compute_parametric_var(abs_res["total_variance"], confidence_level=0.95, horizon_days=1)
        var_99 = RiskDecompositionEngine.compute_parametric_var(abs_res["total_variance"], confidence_level=0.99, horizon_days=1)

        return {
            "symbols": symbols,
            "weights": dict(zip(symbols, weights.tolist())),
            "abs_res": abs_res,
            "rc_res": rc_res,
            "var_1d_95": var_95,
            "var_1d_99": var_99,
        }

    @staticmethod
    def check_budget(risk: dict, budget: dict) -> list:
        """
        budget keys (all optional — only configured keys are checked):
            max_annualized_vol: float — cap on abs_res['annualized_vol']
            max_var_1d_95: float — cap on var_1d_95['var_pct']
            max_specific_var_ratio: float — cap on abs_res['specific_var_ratio'] (concentration
                in single-stock, non-diversifiable risk)
            max_single_stock_risk_pct: float — cap on any one holding's share of total portfolio
                risk (rc_res['risk_contribution_pct']), checked per holding
            max_factor_exposure: {factor_name: float} — cap on abs(portfolio_exposure[factor]),
                checked per factor

        Returns a list of {'check', 'value', 'limit', 'breached'} records, one per configured
        scalar check plus one per holding / per factor for the two per-item checks above.
        """
        results = []
        abs_res = risk["abs_res"]
        rc_res = risk["rc_res"]

        def _add(check, value, limit, breached):
            results.append({"check": check, "value": value, "limit": limit, "breached": breached})

        if "max_annualized_vol" in budget:
            limit = budget["max_annualized_vol"]
            value = abs_res["annualized_vol"]
            _add("annualized_vol", value, limit, value > limit)

        if "max_var_1d_95" in budget:
            limit = budget["max_var_1d_95"]
            value = risk["var_1d_95"]["var_pct"]
            _add("var_1d_95", value, limit, value > limit)

        if "max_specific_var_ratio" in budget:
            limit = budget["max_specific_var_ratio"]
            value = abs_res["specific_var_ratio"]
            _add("specific_var_ratio", value, limit, value > limit)

        if "max_single_stock_risk_pct" in budget:
            limit = budget["max_single_stock_risk_pct"]
            for symbol, pct in rc_res["risk_contribution_pct"].items():
                _add(f"single_stock_risk_pct[{symbol}]", pct, limit, pct > limit)

        if "max_factor_exposure" in budget:
            for factor, limit in budget["max_factor_exposure"].items():
                if factor not in abs_res["portfolio_exposure"]:
                    raise ValueError(f"max_factor_exposure references unknown factor '{factor}'")
                value = abs_res["portfolio_exposure"][factor]
                _add(f"factor_exposure[{factor}]", value, limit, abs(value) > limit)

        return results
