"""
Periodic portfolio risk monitoring — step 7 of the passive factor-allocation workflow.

Recomputes the current risk profile of a saved portfolio (holdings.json) against whatever is
currently in factor_risk.db, and checks it against a risk budget (risk_budget.json). Run this
after every `python main.py` refresh instead of relying on remembering to reopen app.py and
eyeball the numbers.

Exits with a non-zero status if any budget check is breached, so it can be wired into a cron job
or CI-style alert (e.g. `python monitor_portfolio.py || mail -s "risk breach" you@example.com`).

Usage:
    python monitor_portfolio.py --holdings holdings.json --budget risk_budget.json

To run this automatically, add a cron entry — e.g. every weekday at 7am, refresh the pipeline
then check the budget:
    0 7 * * 1-5 cd /path/to/AlphaRisk-Engine && python main.py && python monitor_portfolio.py --holdings holdings.json --budget risk_budget.json
"""
import argparse
import json
import sys

from src.db_client import RiskDatabase
from src.portfolio_monitor import PortfolioMonitor


def _load_json(path: str, label: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read {label} file '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Check a portfolio's current risk against a risk budget.")
    parser.add_argument("--holdings", default="holdings.json", help="JSON file of {symbol: weight}.")
    parser.add_argument("--budget", default="risk_budget.json", help="JSON file of risk-budget limits.")
    parser.add_argument("--db", default="factor_risk.db")
    args = parser.parse_args()

    holdings = _load_json(args.holdings, "holdings")
    budget = _load_json(args.budget, "risk budget")
    if not holdings:
        print(f"Holdings file '{args.holdings}' is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        with RiskDatabase(args.db) as db:
            risk = PortfolioMonitor.compute_portfolio_risk(db, holdings)
    except ValueError as e:
        print(f"Could not compute portfolio risk: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        checks = PortfolioMonitor.check_budget(risk, budget)
    except ValueError as e:
        print(f"Invalid risk budget: {e}", file=sys.stderr)
        sys.exit(1)

    if not checks:
        print("No budget checks configured — nothing to compare against. See risk_budget.example.json.")
        return

    print(f"Portfolio: {', '.join(risk['symbols'])}")
    print(f"Annualized volatility: {risk['abs_res']['annualized_vol']:.2%}")
    print(f"1-day 95% VaR: {risk['var_1d_95']['var_pct']:.2%}   1-day 99% VaR: {risk['var_1d_99']['var_pct']:.2%}")
    print()

    any_breach = False
    for c in checks:
        status = "BREACH" if c["breached"] else "ok"
        any_breach = any_breach or c["breached"]
        print(f"  [{status:>6}] {c['check']:<35} value={c['value']:+.4f}  limit={c['limit']:.4f}")

    if any_breach:
        print("\n⚠️  One or more risk-budget checks were breached — review the portfolio before the next rebalance.")
        sys.exit(1)
    else:
        print("\nAll risk-budget checks passed.")


if __name__ == "__main__":
    main()
