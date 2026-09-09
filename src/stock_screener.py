import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Named factor-tilt presets over the Carhart six factors, expressed as (min_inclusive, max_inclusive)
# bounds — either side may be None for "unbounded". These are directional starting points backed by
# the long-run academic factor-premium literature (value, quality, low-beta), not a promise that any
# of them will outperform going forward — see the README's "known limitations" section. Screening on
# them narrows the universe to a candidate pool; it does not rank stocks by expected return.
STYLE_PRESETS = {
    "value": {"HML": (0.3, None)},
    "quality": {"RMW": (0.2, None)},
    "low_vol": {"MKT": (None, 0.9)},
    "momentum": {"MOM": (0.2, None)},
    "defensive": {"MKT": (None, 0.9), "RMW": (0.1, None)},
}


def _tighten_min(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _tighten_max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


class StockScreener:
    """Filters and ranks the factor-exposure universe produced by main.py's regression pipeline.

    This is a screening tool, not a signal generator: it narrows several hundred stocks down to a
    candidate pool matching a chosen factor tilt (e.g. "value + quality"), so a human can do the
    final judgment call — step 3 of the passive factor-allocation workflow described in the
    README. It ranks by *exposure to a factor*, never by predicted future return.
    """

    @staticmethod
    def merge_factor_bounds(*bound_dicts: dict) -> dict:
        """Merges several {factor_name: (min, max)} dicts into one, tightening the bound on any
        factor that appears in more than one (max of the mins, min of the maxes) rather than
        letting a later dict silently overwrite an earlier one."""
        merged: dict = {}
        for bounds in bound_dicts:
            for factor, (lo, hi) in bounds.items():
                cur_lo, cur_hi = merged.get(factor, (None, None))
                merged[factor] = (_tighten_min(cur_lo, lo), _tighten_max(cur_hi, hi))
        return merged

    @staticmethod
    def resolve_style(style_names: list) -> dict:
        """Merges one or more named STYLE_PRESETS into a single factor-bounds dict."""
        unknown = [s for s in style_names if s not in STYLE_PRESETS]
        if unknown:
            raise ValueError(f"Unknown style preset(s) {unknown}. Known presets: {sorted(STYLE_PRESETS)}")
        return StockScreener.merge_factor_bounds(*(STYLE_PRESETS[s] for s in style_names))

    @staticmethod
    def compute_composite_score(df: pd.DataFrame, factor_weights: dict) -> pd.Series:
        """Combines several factor exposures into a single ranking score.

        Each factor is z-scored *within the given df* (mean 0, std 1 across whichever rows are
        passed in — typically the already-filtered candidate pool, not the whole universe, so the
        scale reflects "how extreme is this stock on this factor, relative to the other survivors"
        rather than the full S&P 500). The z-scores are then combined with factor_weights and
        summed — e.g. {'HML': 1.0, 'RMW': 1.0} ranks candidates by "value + quality combined",
        each factor contributing comparably regardless of its own natural scale.

        This is still a ranking by *factor exposure*, not a predicted return — a higher composite
        score means "more exposed to the requested factor tilt," not "expected to perform better."
        """
        if df.empty:
            return pd.Series(dtype=float, index=df.index)
        score = pd.Series(0.0, index=df.index)
        for factor, weight in factor_weights.items():
            if factor not in df.columns:
                raise ValueError(f"compute_composite_score references unknown factor column '{factor}'")
            col = df[factor].astype(float)
            std = col.std(ddof=0)
            z = (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=df.index)
            score = score + z * weight
        return score

    @staticmethod
    def screen(
        exposures_df: pd.DataFrame,
        factor_bounds: dict = None,
        min_r_squared: float = 0.0,
        max_idiosyncratic_var: float = None,
        exclude_symbols: list = None,
        max_per_sector: int = None,
        sort_by: str = None,
        ascending: bool = False,
        composite_weights: dict = None,
        top_n: int = None,
    ) -> pd.DataFrame:
        """
        Args:
            exposures_df: output of RiskDatabase.load_exposures_wide(), optionally merged with
                securities metadata (sec_name, sector) — the same shape app.py loads.
            factor_bounds: {factor_name: (min_inclusive_or_None, max_inclusive_or_None)}.
            min_r_squared: drop stocks whose six-factor regression fit is below this — a low R^2
                means the loadings themselves are a noisy, unreliable read on this stock's style.
            max_idiosyncratic_var: optional cap on single-stock idiosyncratic variance, to exclude
                names whose risk is dominated by stock-specific events the factor model can't see.
            exclude_symbols: tickers to drop before filtering (e.g. the 'SPY' benchmark row).
            max_per_sector: cap how many names from any one GICS sector survive, applied after
                ranking — keeps the candidate pool from being dominated by one crowded sector.
            sort_by: a column (factor name, 'r_squared', etc.) to rank the surviving rows by.
                Ignored if composite_weights is given.
            composite_weights: if given, overrides sort_by — ranks by a single composite score
                (see compute_composite_score) combining several factors' z-scores, computed on
                the pool that survives the filters above. The score is added as a 'composite_score'
                column on the returned DataFrame.
            top_n: keep only the top N rows after sorting and the per-sector cap.

        Returns a filtered/sorted copy of exposures_df with its original columns preserved.
        """
        df = exposures_df.copy()

        if exclude_symbols:
            df = df[~df["symbol"].isin(exclude_symbols)]

        if min_r_squared > 0:
            if "r_squared" not in df.columns:
                raise ValueError("min_r_squared filter requires an 'r_squared' column")
            df = df[df["r_squared"] >= min_r_squared]

        if max_idiosyncratic_var is not None:
            if "idiosyncratic_var" not in df.columns:
                raise ValueError("max_idiosyncratic_var filter requires an 'idiosyncratic_var' column")
            df = df[df["idiosyncratic_var"] <= max_idiosyncratic_var]

        for factor, (lo, hi) in (factor_bounds or {}).items():
            if factor not in df.columns:
                raise ValueError(f"factor_bounds references unknown factor column '{factor}'")
            if lo is not None:
                df = df[df[factor] >= lo]
            if hi is not None:
                df = df[df[factor] <= hi]

        if composite_weights:
            df = df.copy()
            df["composite_score"] = StockScreener.compute_composite_score(df, composite_weights)
            df = df.sort_values("composite_score", ascending=ascending)
        elif sort_by is not None:
            if sort_by not in df.columns:
                raise ValueError(f"sort_by references unknown column '{sort_by}'")
            df = df.sort_values(sort_by, ascending=ascending)

        if max_per_sector is not None:
            if "sector" not in df.columns:
                raise ValueError("max_per_sector filter requires a 'sector' column")
            # Relies on df already being in final rank order (post sort_by above): cumcount() on
            # each sector group in that order keeps exactly the top max_per_sector-ranked names.
            df = df[df.groupby("sector").cumcount() < max_per_sector]

        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index(drop=True)
