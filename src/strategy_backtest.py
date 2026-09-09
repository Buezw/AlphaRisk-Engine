import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MomentumBacktester:
    """
    Backtests a simple weekly-rebalanced cross-sectional momentum strategy: rank the universe by
    trailing price momentum as of each week's first trading day, hold the top-K names equal-weighted
    for that week, then re-rank.

    This is a starting point for testing whether a specific, simple rule would have beaten the
    benchmark historically — it is not a profit guarantee. Past performance is not predictive of
    future returns, and the price history available here is a single, possibly unrepresentative,
    ~1-year window (see main.py's rolling LOOKBACK_YEARS / prune_prices_before), not a multi-cycle
    sample.
    """

    @staticmethod
    def build_weekly_execution_dates(dates: pd.DatetimeIndex) -> list:
        """Picks the first available trading day of each ISO calendar week as that week's
        rebalance/execution date (a stand-in for "every Monday" when Monday itself is a holiday)."""
        dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
        iso = dates.isocalendar()
        key = list(zip(iso['year'], iso['week']))
        df = pd.DataFrame({'date': dates, 'key': key})
        first_per_week = df.groupby('key')['date'].min()
        return sorted(first_per_week.tolist())

    @staticmethod
    def run(
        prices_wide: pd.DataFrame,
        top_k: int = 10,
        lookback_weeks: int = 12,
        cost_bps: float = 0.0,
        require_full_history: bool = True,
    ) -> dict:
        """
        Args:
            prices_wide: DatetimeIndex-indexed DataFrame, one column per symbol, adjusted close prices.
            top_k: number of names to hold each week.
            lookback_weeks: trailing window (in weeks) used to rank momentum.
            cost_bps: round-trip transaction cost, in basis points of the traded value, charged on
                every unit of weekly turnover (a name entering or leaving the top-K). 0 = no cost
                modeling (matches this function's original behavior).
            require_full_history: if True (default), a symbol must already have a valid price on
                this DataFrame's very first date to ever be eligible for selection. This is a
                partial mitigation for survivorship bias — it excludes names that only entered the
                available universe partway through the window (e.g. because they were added to the
                S&P 500 index, which this project's data fetch keys off of, sometime after the
                window started, often *because* of a run-up in price that would otherwise inflate a
                momentum backtest). It does NOT fix the other half of survivorship bias — names that
                were removed from today's index and so were never fetched at all — which would need
                a point-in-time historical constituent list (not freely available; see
                fetch_backtest_history.py's module docstring).

        Returns:
            {'weekly_returns': DataFrame indexed by the date the week's return realizes on, with
                columns 'return' (gross), 'net_return' (after cost_bps), 'turnover', 'n_selected',
             'selected_history': list of {'date', 'symbols'} showing what was picked each week}
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if lookback_weeks < 1:
            raise ValueError("lookback_weeks must be >= 1")
        if cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")

        prices_wide = prices_wide.sort_index()

        if require_full_history and not prices_wide.empty:
            eligible_cols = prices_wide.columns[prices_wide.iloc[0].notna()]
            excluded = len(prices_wide.columns) - len(eligible_cols)
            if excluded:
                logger.info(
                    f"require_full_history: excluding {excluded} symbol(s) without a valid price "
                    f"on {prices_wide.index[0].date()} (the window's first date)."
                )
            prices_wide = prices_wide[eligible_cols]

        exec_dates = MomentumBacktester.build_weekly_execution_dates(prices_wide.index)
        if len(exec_dates) < lookback_weeks + 2:
            raise ValueError(
                f"Not enough weekly history to backtest: need at least {lookback_weeks + 2} "
                f"weeks, have {len(exec_dates)}."
            )

        weekly_rows = []
        selected_history = []
        prev_selected = set()

        for i in range(lookback_weeks, len(exec_dates) - 1):
            signal_date = exec_dates[i]
            lookback_date = exec_dates[i - lookback_weeks]
            next_date = exec_dates[i + 1]

            # Momentum signal: trailing return from lookback_date to signal_date. Only symbols with
            # a valid price at both ends are eligible — a missing datapoint means "we don't actually
            # know this stock's momentum", not "flat", so it must not silently rank as 0%.
            p0 = prices_wide.loc[lookback_date]
            p1 = prices_wide.loc[signal_date]
            valid = p0.notna() & p1.notna() & (p0 > 0)
            momentum = ((p1 - p0) / p0)[valid]

            if momentum.empty:
                weekly_rows.append({'date': next_date, 'return': 0.0, 'net_return': 0.0, 'turnover': 0.0, 'n_selected': 0})
                selected_history.append({'date': signal_date, 'symbols': []})
                prev_selected = set()
                continue

            top = momentum.sort_values(ascending=False).head(top_k)
            selected = list(top.index)

            # Realized return for the week just held (signal_date close -> next_date close),
            # equal-weighted across the selected names. Any name missing next_date's price is
            # dropped and the rest re-weighted rather than silently counted as a 0% return.
            p_start = prices_wide.loc[signal_date, selected]
            p_end = prices_wide.loc[next_date, selected]
            held_valid = p_start.notna() & p_end.notna() & (p_start > 0)
            realized = ((p_end - p_start) / p_start)[held_valid]

            week_return = float(realized.mean()) if not realized.empty else 0.0

            # Turnover: fraction of the (equal-weighted) portfolio that had to be sold and rebought
            # this week. Standard definition — half the sum of absolute weight changes — collapses
            # to (names replaced) / top_k when every held name carries the same 1/top_k weight.
            n_changed = len(set(selected) - prev_selected)
            turnover = n_changed / top_k
            net_return = week_return - turnover * (cost_bps / 10000.0)

            weekly_rows.append({
                'date': next_date, 'return': week_return, 'net_return': net_return,
                'turnover': turnover, 'n_selected': int(len(realized)),
            })
            selected_history.append({'date': signal_date, 'symbols': selected})
            prev_selected = set(selected)

        returns_df = pd.DataFrame(weekly_rows).set_index('date')
        return {'weekly_returns': returns_df, 'selected_history': selected_history}

    @staticmethod
    def generate_current_picks(
        prices_wide: pd.DataFrame, top_k: int = 10, lookback_weeks: int = 12, require_full_history: bool = True
    ) -> dict:
        """
        Live counterpart to run(): ranks the universe by trailing momentum as of the most recent
        available trading day (rather than replaying history) and returns this week's target list.

        This still carries the same caveats as the backtest — no profit guarantee, and whatever
        biases (survivorship, no transaction costs) affected the backtested numbers apply here too.
        See run()'s docstring for what require_full_history does and does not fix.
        """
        prices_wide = prices_wide.sort_index()
        if prices_wide.empty:
            raise ValueError("prices_wide is empty")

        if require_full_history:
            eligible_cols = prices_wide.columns[prices_wide.iloc[0].notna()]
            prices_wide = prices_wide[eligible_cols]

        latest_date = prices_wide.index.max()
        lookback_target = latest_date - pd.Timedelta(weeks=lookback_weeks)
        available = prices_wide.index[prices_wide.index <= lookback_target]
        if available.empty:
            raise ValueError(
                f"Not enough history before {lookback_target.date()} to compute a "
                f"{lookback_weeks}-week momentum signal."
            )
        lookback_date = available.max()

        p0 = prices_wide.loc[lookback_date]
        p1 = prices_wide.loc[latest_date]
        valid = p0.notna() & p1.notna() & (p0 > 0)
        momentum = ((p1 - p0) / p0)[valid]
        if momentum.empty:
            raise ValueError("No symbols have valid prices at both the signal date and lookback date.")

        top = momentum.sort_values(ascending=False).head(top_k)
        target_weight = 1.0 / len(top)

        return {
            'signal_date': latest_date,
            'lookback_date': lookback_date,
            'picks': pd.DataFrame({
                'momentum_return': top,
                'target_weight': target_weight,
            }),
        }

    @staticmethod
    def performance_stats(weekly_returns: pd.Series, periods_per_year: int = 52) -> dict:
        """Summary stats for a series of periodic (weekly) returns."""
        r = pd.Series(weekly_returns).dropna()
        if r.empty:
            return {}

        cumulative = (1 + r).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)
        n_years = len(r) / periods_per_year
        annualized_return = float(cumulative.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
        annualized_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
        sharpe_ratio = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        max_drawdown = float(drawdown.min())
        win_rate = float((r > 0).mean())

        return {
            'n_periods': int(len(r)),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'equity_curve': cumulative,
        }
