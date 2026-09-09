"""
Factor-based stock screening CLI — step 3 of the passive factor-allocation workflow: narrows the
S&P 500 universe (already regressed by main.py's factor pipeline) down to a candidate pool that
matches one or more chosen style tilts, for a human to review and finalize by hand.

This ranks stocks by *exposure to a factor*, never by predicted future return — it is a filter
over already-computed factor loadings, not a signal generator. See README's known limitations.

Usage:
    python screen_stocks.py --style value quality --min-r2 0.3 --max-per-sector 3 --top 30
    python screen_stocks.py --factor HML 0.3 None --factor RMW 0.2 None --top 20 --out picks.csv
"""
import argparse
import sys

import pandas as pd

from src.db_client import RiskDatabase
from src.stock_screener import STYLE_PRESETS, StockScreener


def _parse_bound(s: str):
    return None if s.lower() == "none" else float(s)


def main():
    parser = argparse.ArgumentParser(description="Screen the factor-exposure universe by style tilt.")
    parser.add_argument(
        "--style", nargs="+", choices=sorted(STYLE_PRESETS), default=[],
        help="One or more named style presets to combine (overlapping bounds are tightened, not OR'd)."
    )
    parser.add_argument(
        "--factor", nargs=3, action="append", default=[], metavar=("NAME", "MIN", "MAX"),
        help="Custom factor bound, e.g. --factor HML 0.3 None. Repeatable. 'None' = unbounded on that side."
    )
    parser.add_argument("--min-r2", type=float, default=0.2, help="Minimum regression fit (R^2) to keep a stock.")
    parser.add_argument("--max-idio-var", type=float, default=None, help="Maximum idiosyncratic variance to keep a stock.")
    parser.add_argument("--max-per-sector", type=int, default=None, help="Cap candidates per GICS sector.")
    parser.add_argument("--sort-by", default=None, help="Column to rank by (default: the first bound's factor).")
    parser.add_argument("--top", type=int, default=30, help="Keep only the top N results.")
    parser.add_argument("--db", default="factor_risk.db", help="Path to the SQLite database.")
    parser.add_argument("--out", default=None, help="Optional CSV path to also write the results to.")
    args = parser.parse_args()

    if not args.style and not args.factor:
        parser.error("Provide at least one --style preset or --factor bound.")

    try:
        bounds = StockScreener.merge_factor_bounds(
            StockScreener.resolve_style(args.style),
            {name: (_parse_bound(lo), _parse_bound(hi)) for name, lo, hi in args.factor},
        )
    except ValueError as e:
        parser.error(str(e))

    sort_by = args.sort_by or next(iter(bounds))

    try:
        with RiskDatabase(args.db) as db:
            exposures_df = db.load_exposures_wide()
            sec_df = pd.read_sql_query("SELECT symbol, sec_name, sector FROM securities", db.conn)
    except Exception as e:
        print(f"Failed to load {args.db}: {e}", file=sys.stderr)
        sys.exit(1)

    if exposures_df.empty:
        print(f"{args.db} has no factor exposures yet — run `python main.py` first.", file=sys.stderr)
        sys.exit(1)

    exposures_df = pd.merge(exposures_df, sec_df, on="symbol", how="left")

    try:
        result = StockScreener.screen(
            exposures_df,
            factor_bounds=bounds,
            min_r_squared=args.min_r2,
            max_idiosyncratic_var=args.max_idio_var,
            exclude_symbols=["SPY"],
            max_per_sector=args.max_per_sector,
            sort_by=sort_by,
            top_n=args.top,
        )
    except ValueError as e:
        print(f"Screen failed: {e}", file=sys.stderr)
        sys.exit(1)

    if result.empty:
        print("No stocks matched the given filters.")
        return

    display_cols = ["symbol", "sec_name", "sector"] + list(bounds.keys()) + ["r_squared"]
    display_cols = [c for c in display_cols if c in result.columns]
    print(result[display_cols].to_string(index=False))
    print(f"\n{len(result)} candidates matched (sorted by {sort_by}).")

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
