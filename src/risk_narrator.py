"""
risk_narrator.py
------------------
Translates the numeric output of the six-factor risk decomposition into
human-readable diagnostic text. No LLM involved — a pure rule engine that
runs offline, is fast, deterministic, and auditable.

Design principles:
1. Every number needs a "reference frame" (is this high/medium/low) — never
   hand the user a bare percentage with nothing to compare it to.
2. Every conclusion should explain "what this means", not just restate the number.
3. Layered disclosure: lead with the single most important takeaway, then details.
"""

from dataclasses import dataclass
from typing import Literal

FACTOR_LABELS = {
    'MKT': 'Market',
    'Mkt-RF': 'Market',
    'SMB': 'Size',
    'HML': 'Value',
    'RMW': 'Profitability',
    'CMA': 'Investment',
    'MOM': 'Momentum',
    'WML': 'Momentum',
}

FACTOR_MEANING = {
    'SMB': ('small-cap stocks', 'large-cap stocks'),
    'HML': ('value stocks (high book-to-market)', 'growth stocks (low book-to-market)'),
    'RMW': ('high-profitability companies', 'low-profitability companies'),
    'CMA': ('conservative-investment companies', 'aggressive-expansion companies'),
    'MOM': ('recent winners (strong momentum)', 'recent losers (weak momentum)'),
    'WML': ('recent winners (strong momentum)', 'recent losers (weak momentum)'),
}


def _level(value: float, thresholds: tuple, labels: tuple) -> str:
    """Generic bucketing helper: maps a value to a Low/Medium/High-style label based on thresholds."""
    for t, label in zip(thresholds, labels[:-1]):
        if value < t:
            return label
    return labels[-1]


def _factor_display_name(col: str) -> str:
    return FACTOR_LABELS.get(col, col)


@dataclass
class DiagnosisLine:
    icon: str
    level: Literal["good", "warning", "danger", "info"]
    headline: str
    detail: str


# ----------------------------------------------------------------
# Active risk (relative to SPY) diagnostics
# ----------------------------------------------------------------

def diagnose_active_risk(act_res: dict, factor_cols: list, n_holdings: int) -> list[DiagnosisLine]:
    lines = []

    te = act_res['tracking_error']
    factor_ratio = act_res['active_factor_ratio']
    specific_ratio = act_res['active_specific_ratio']
    tilts = act_res['active_tilts']

    # --- 1. Tracking error level: give an industry reference frame ---
    te_label = _level(
        te,
        thresholds=(0.02, 0.05, 0.10),
        labels=("Very Low", "Moderate", "Elevated", "Very High")
    )
    te_context = {
        "Very Low": "close to the level of an index-enhanced strategy (most enhanced-index funds run 1%-3% TE)",
        "Moderate": "a common range for typical actively managed funds (most active funds run 3%-6% TE)",
        "Elevated": "already beyond the common range for most active funds — performance may diverge noticeably from the benchmark",
        "Very High": "close to the level of a focused/concentrated strategy, amplifying the odds of both beating and lagging the benchmark in the short term",
    }
    lines.append(DiagnosisLine(
        icon="📏",
        level="warning" if te_label in ("Elevated", "Very High") else "info",
        headline=f"Annualized tracking error is {te:.1%}, which is \"{te_label}\"",
        detail=f"{te_context[te_label]}. Tracking error measures the annualized volatility of the portfolio's return "
               f"relative to SPY — it doesn't say whether the portfolio wins or loses, only how sharply it diverges from the benchmark."
    ))

    # --- 2. Source of risk: style tilts vs. stock-picking concentration ---
    if factor_ratio > specific_ratio:
        dominant = "style factor exposure (Style Tilts)"
        implication = "The portfolio's active risk mainly comes from bets on systematic styles (e.g. size, value, momentum), rather than from which specific stocks were chosen."
    else:
        dominant = "single-stock concentration (Stock Picking)"
        implication = "The portfolio's active risk mainly comes from concentration in specific holdings rather than systematic style bets — closer to a \"stock-picking\" approach than a \"style-rotation\" one."

    lines.append(DiagnosisLine(
        icon="🎯",
        level="info",
        headline=f"The main source of active risk is \"{dominant}\" "
                  f"({max(factor_ratio, specific_ratio):.0%} of tracking-error variance)",
        detail=implication
    ))

    # --- 3. Most prominent factor exposure ---
    if tilts:
        dominant_factor = max(tilts, key=lambda k: abs(tilts[k]))
        dominant_value = tilts[dominant_factor]
        factor_name = _factor_display_name(dominant_factor)

        if dominant_factor in FACTOR_MEANING:
            high_side, low_side = FACTOR_MEANING[dominant_factor]
            direction_text = f"leans more toward {high_side}" if dominant_value > 0 else f"leans more toward {low_side}"
        else:
            direction_text = "overweight" if dominant_value > 0 else "underweight"

        magnitude_label = _level(
            abs(dominant_value),
            thresholds=(0.15, 0.35),
            labels=("slight", "notable", "strong")
        )

        lines.append(DiagnosisLine(
            icon="⚖️",
            level="warning" if magnitude_label == "strong" else "info",
            headline=f"Relative to SPY, the portfolio has a {magnitude_label} deviation on the \"{factor_name}\" factor ({dominant_value:+.2f})",
            detail=f"This means, compared with the S&P 500, the portfolio {direction_text}. "
                   f"If this factor underperforms going forward, the odds of lagging SPY rise accordingly — and vice versa."
        ))

    # --- 4. Concentration warning ---
    if n_holdings <= 3:
        lines.append(DiagnosisLine(
            icon="⚠️",
            level="danger",
            headline=f"Only {n_holdings} stocks currently held — idiosyncratic risk is not well diversified",
            detail="With a small number of holdings, an idiosyncratic event in any one stock (an earnings miss, a regulatory probe, etc.) "
                   "can meaningfully hit the whole portfolio — a risk the six-factor model cannot explain or hedge."
        ))

    return lines


# ----------------------------------------------------------------
# Absolute risk diagnostics
# ----------------------------------------------------------------

def diagnose_absolute_risk(abs_res: dict, factor_cols: list, n_holdings: int) -> list[DiagnosisLine]:
    lines = []

    vol = abs_res['annualized_vol']
    factor_ratio = abs_res['factor_var_ratio']
    specific_ratio = abs_res['specific_var_ratio']

    # --- 1. Volatility level ---
    vol_label = _level(
        vol,
        thresholds=(0.12, 0.20, 0.30),
        labels=("Low", "Moderate", "Elevated", "Very High")
    )
    vol_context = {
        "Low": "below the S&P 500's historical average volatility (roughly 15%-18%) — an overall defensive tilt",
        "Moderate": "close to the S&P 500's historical average volatility",
        "Elevated": "notably above the S&P 500's historical average volatility — the portfolio is taking on more risk in exchange for potentially higher return",
        "Very High": "far above the market average — worth confirming this matches your intended risk tolerance",
    }
    lines.append(DiagnosisLine(
        icon="📊",
        level="warning" if vol_label in ("Elevated", "Very High") else "info",
        headline=f"Portfolio annualized volatility is {vol:.1%}, which is \"{vol_label}\"",
        detail=vol_context[vol_label]
    ))

    # --- 2. Systematic vs. idiosyncratic risk share ---
    if specific_ratio > 0.6:
        lines.append(DiagnosisLine(
            icon="🎲",
            level="danger",
            headline=f"Idiosyncratic risk share is as high as {specific_ratio:.0%} — the six-factor model has limited explanatory power here",
            detail="This means most of the portfolio's volatility comes from idiosyncratic, single-stock events (earnings, industry news), "
                   "rather than the common trends captured by systematic factors like size, value, and momentum. "
                   "Increasing the number of holdings and diversifying across sectors usually brings this ratio down."
        ))
    elif specific_ratio > 0.35:
        lines.append(DiagnosisLine(
            icon="🎲",
            level="warning",
            headline=f"Idiosyncratic risk share is {specific_ratio:.0%}, a moderate level",
            detail="A meaningful chunk of the portfolio's risk still can't be explained by the six factors. "
                   "If the goal is pure style exposure (rather than stock-picking alpha), consider diversifying holdings a bit further."
        ))
    else:
        lines.append(DiagnosisLine(
            icon="✅",
            level="good",
            headline=f"Systematic factor risk share is {factor_ratio:.0%} — the portfolio's risk is mostly explained by the six factors",
            detail="Most of the portfolio's volatility can be attributed to the well-known systematic style factors — "
                   "market, size, value, profitability, investment, and momentum — with relatively limited impact from single-stock events."
        ))

    return lines


# ----------------------------------------------------------------
# Per-stock style tags & factor narrative (replaces "cramming 6 beta
# columns into the table")
# ----------------------------------------------------------------

# Per factor: positive-exposure label, negative-exposure label; hidden when neutral
STYLE_TAGS = {
    'MKT':     ('High Beta', 'Low Beta'),
    'Mkt-RF':  ('High Beta', 'Low Beta'),
    'SMB':     ('Small-Cap Tilt', 'Large-Cap Tilt'),
    'HML':     ('Value Tilt', 'Growth Tilt'),
    'RMW':     ('High Profitability', 'Low Profitability'),
    'CMA':     ('Conservative Investment', 'Aggressive Investment'),
    'MOM':     ('Strong Momentum', 'Reversal / Weak Momentum'),
    'WML':     ('Strong Momentum', 'Reversal / Weak Momentum'),
}

# Threshold for a "significant exposure": only tagged once the absolute value exceeds this, to avoid noise
_TAG_THRESHOLD = 0.15


def build_style_tags(row: dict, factor_cols: list, max_tags: int = 2) -> str:
    """
    Generates a short style-tag string for a row of per-stock data, e.g. "Strong Momentum · High Profitability".
    Only picks the max_tags factors with the strongest exposure, to avoid tag pile-up.
    MKT is usually close to 1 for every stock, so it's excluded from tag selection by default
    (unless it deviates from 1 noticeably).
    """
    candidates = []
    for col in factor_cols:
        if col not in STYLE_TAGS:
            continue
        val = row.get(col)
        if val is None or pd_isna(val):
            continue

        # Special-case the market factor: judge by deviation from 1, not deviation from 0
        if col in ('MKT', 'Mkt-RF'):
            deviation = val - 1.0
            if abs(deviation) < _TAG_THRESHOLD:
                continue
            pos_label, neg_label = STYLE_TAGS[col]
            candidates.append((abs(deviation), pos_label if deviation > 0 else neg_label))
            continue

        if abs(val) < _TAG_THRESHOLD:
            continue
        pos_label, neg_label = STYLE_TAGS[col]
        candidates.append((abs(val), pos_label if val > 0 else neg_label))

    if not candidates:
        return "Style-Neutral"

    candidates.sort(key=lambda x: -x[0])
    top_tags = [label for _, label in candidates[:max_tags]]
    return " · ".join(top_tags)


def build_active_tilt_summary(active_tilts: dict, factor_cols: list, max_tags: int = 2) -> str:
    """
    Condenses the "active factor tilt" (Δβ = Bᵀ Δw) into a one-line takeaway, e.g.
    "Relative to the S&P 500, this portfolio's active style bet is 'Strong Momentum'."
    """
    tags = build_style_tags(active_tilts, factor_cols, max_tags=max_tags)
    if tags == "Style-Neutral":
        return "Relative to the S&P 500, this portfolio has no notable deviation across the six style factors — its active style is close to neutral."
    return f"Relative to the S&P 500, this portfolio's active style bet is \"{tags}\"."


def build_portfolio_exposure_summary(portfolio_exposure: dict, factor_cols: list, max_tags: int = 2) -> str:
    """
    Condenses the "portfolio-level six-factor exposure" (B_p = B^T w) into a one-line profile, e.g.
    "This portfolio is essentially a high-momentum, low-value portfolio."
    Reuses the per-stock style-tag logic, just with the weighted portfolio-level beta as input.
    """
    tags = build_style_tags(portfolio_exposure, factor_cols, max_tags=max_tags)
    if tags == "Style-Neutral":
        return "This portfolio's exposure is balanced across the six style factors, with no single dominant style bet."
    return f"This portfolio is essentially a \"{tags}\" portfolio."


def pd_isna(val) -> bool:
    """Avoids a direct pandas dependency in this module, to keep it lightweight."""
    try:
        return val != val  # the classic NaN test: NaN != NaN
    except Exception:
        return val is None


def _factor_bar_color(col: str, val: float) -> str:
    """The color a given factor's β value should get — blue/red/gray semantics stay consistent with the bar charts and the Δβ narrative."""
    if col in ('MKT', 'Mkt-RF'):
        deviation = val - 1.0
        if abs(deviation) < 0.05:
            return _NEUTRAL_COLOR
        return _POS_COLOR if deviation > 0 else _NEG_COLOR
    if abs(val) < _TAG_THRESHOLD:
        return _NEUTRAL_COLOR
    return _POS_COLOR if val > 0 else _NEG_COLOR


def factor_bar_colors(row: dict, factor_cols: list) -> list:
    """Color list for the horizontal bar chart, matching build_factor_narrative's text colors one-to-one."""
    return [_factor_bar_color(c, row.get(c)) for c in factor_cols if c in STYLE_TAGS]


def build_factor_narrative(symbol: str, row: dict, factor_cols: list, subject_label: str = "this stock") -> str:
    """
    Generates the full factor-exposure narrative HTML for a single stock (or for the portfolio as a
    whole, by passing subject_label="this portfolio") — used in the detail drill-down, meant to be
    rendered via st.markdown(..., unsafe_allow_html=True).
    Colors match the bar chart: blue = positive exposure, red = negative exposure, gray = near-neutral.
    `row` just needs to be a {factor_name: beta} dict — the portfolio-level exposure B_p has exactly
    the same shape as a single stock's betas, so this logic can be reused directly without a separate
    portfolio-level implementation.
    """
    rows = []
    for col in factor_cols:
        if col not in STYLE_TAGS:
            continue
        val = row.get(col)
        if val is None or pd_isna(val):
            continue

        pos_label, neg_label = STYLE_TAGS[col]
        factor_name = _factor_display_name(col)
        color = _factor_bar_color(col, val)

        if col in ('MKT', 'Mkt-RF'):
            deviation = val - 1.0
            if abs(deviation) < 0.05:
                detail = "Its swings track the broad market closely."
            elif deviation > 0:
                magnitude = "notably" if deviation > 0.3 else "slightly"
                detail = f"{magnitude} above average market volatility — {subject_label} tends to amplify moves when the market rises or falls."
            else:
                magnitude = "notably" if deviation < -0.3 else "slightly"
                detail = f"{magnitude} below average market volatility — a relatively defensive holding."
        else:
            magnitude_label = _level(abs(val), thresholds=(0.15, 0.5), labels=("near-neutral", "mild", "pronounced"))
            if magnitude_label == "near-neutral":
                detail = "Exposure is near-neutral — this style dimension isn't a notable factor here."
            else:
                direction = pos_label if val > 0 else neg_label
                detail = f"{magnitude_label} tilt toward \"<span style='color:{color};'>{direction}</span>\"."

        rows.append(
            f"<div style='height:60px; box-sizing:border-box; display:flex; align-items:center; line-height:1.3; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {color};'>"
            f"<span><b>{factor_name}</b>"
            f" (β=<span style='color:{color}; font-weight:600;'>{val:.2f}</span>): "
            f"{detail}</span>"
            f"</div>"
        )

    return "".join(rows)


# Same color semantics as the bar charts: positive = blue, negative = red, neutral = gray
_POS_COLOR = "#2563eb"
_NEG_COLOR = "#ef4444"
_NEUTRAL_COLOR = "#94a3b8"
# Pie-chart-specific colors, matching go.Pie's marker colors
_FACTOR_RISK_COLOR = "#2563eb"
_SPECIFIC_RISK_COLOR = "#f97316"


def _active_tilt_color(val: float) -> str:
    """The color a given Δβ should get; the threshold matches build_active_tilt_narrative's "near-neutral" judgment."""
    if abs(val) < 0.10:
        return _NEUTRAL_COLOR
    return _POS_COLOR if val > 0 else _NEG_COLOR


def active_tilt_bar_colors(active_tilts: dict, factor_cols: list) -> list:
    """Color list for the active-factor-tilt bar chart, matching build_active_tilt_narrative's text colors one-to-one."""
    return [_active_tilt_color(active_tilts.get(c)) for c in factor_cols if c in STYLE_TAGS]


def build_active_tilt_narrative(active_tilts: dict, factor_cols: list) -> str:
    """
    Generates the per-factor narrative HTML for the portfolio's "active factor tilt relative to SPY"
    (Δβ = Bᵀ Δw) — meant to be rendered via st.markdown(..., unsafe_allow_html=True).
    Colors match the bar chart: blue = overweight/positive, red = underweight/negative, gray = neutral.
    Because Δβ is already a "deviation" (centered at 0, not 1), MKT does not get the special
    deviation-from-1.0 treatment here.
    """
    rows = []
    for col in factor_cols:
        if col not in STYLE_TAGS:
            continue
        val = active_tilts.get(col)
        if val is None or pd_isna(val):
            continue

        pos_label, neg_label = STYLE_TAGS[col]
        factor_name = _factor_display_name(col)

        magnitude_label = _level(abs(val), thresholds=(0.10, 0.25), labels=("near-neutral", "mild", "pronounced"))
        if magnitude_label == "near-neutral":
            color = _NEUTRAL_COLOR
            detail = "Exposure relative to SPY is near-neutral — no notable bet on this dimension."
        else:
            color = _POS_COLOR if val > 0 else _NEG_COLOR
            direction = pos_label if val > 0 else neg_label
            detail = f"{magnitude_label} tilt toward \"<span style='color:{color};'>{direction}</span>\" relative to SPY."

        rows.append(
            f"<div style='height:60px; box-sizing:border-box; display:flex; align-items:center; line-height:1.3; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {color};'>"
            f"<span><b>{factor_name}</b>"
            f" (Δβ=<span style='color:{color}; font-weight:600;'>{val:+.2f}</span>): "
            f"{detail}</span>"
            f"</div>"
        )

    return "".join(rows)


def build_te_pie_narrative(act_res: dict) -> str:
    """
    Generates the narrative HTML for the two slices of the "tracking-error variance decomposition" pie
    (active factor risk / active specific risk); colors match the pie's marker colors (blue/orange).
    """
    factor_ratio = act_res['active_factor_ratio']
    specific_ratio = act_res['active_specific_ratio']

    rows = [
        (
            f"<div style='min-height:120px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_FACTOR_RISK_COLOR};'>"
            f"<span><b style='color:{_FACTOR_RISK_COLOR};'>Active Factor Risk</b>"
            f" ({factor_ratio:.0%} of tracking-error variance): "
            f"comes from style-factor misalignment relative to SPY (over/underweight size, value, profitability, momentum, etc.) — "
            f"can be reduced through style-neutralizing hedges.</span></div>"
        ),
        (
            f"<div style='min-height:120px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_SPECIFIC_RISK_COLOR};'>"
            f"<span><b style='color:{_SPECIFIC_RISK_COLOR};'>Active Specific Risk</b>"
            f" ({specific_ratio:.0%} of tracking-error variance): "
            f"comes from concentration in specific holdings rather than a systematic style bet — can be reduced by diversifying holdings further.</span></div>"
        ),
    ]
    return "".join(rows)


def build_abs_variance_pie_narrative(abs_res: dict) -> str:
    """
    Generates the narrative HTML for the two slices of the "total variance decomposition" pie
    (systematic factor risk / idiosyncratic risk); colors match the pie's marker colors (blue/orange),
    symmetric in structure to build_te_pie_narrative.
    """
    factor_ratio = abs_res['factor_var_ratio']
    specific_ratio = abs_res['specific_var_ratio']

    rows = [
        (
            f"<div style='min-height:120px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_FACTOR_RISK_COLOR};'>"
            f"<span><b style='color:{_FACTOR_RISK_COLOR};'>Systematic Factor Risk</b>"
            f" ({factor_ratio:.0%} of total portfolio variance): "
            f"volatility jointly explained by the six systematic factors — market, size, value, profitability, investment, and momentum — "
            f"driven by the overall market environment and not removable through diversification.</span></div>"
        ),
        (
            f"<div style='min-height:120px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_SPECIFIC_RISK_COLOR};'>"
            f"<span><b style='color:{_SPECIFIC_RISK_COLOR};'>Idiosyncratic Risk</b>"
            f" ({specific_ratio:.0%} of total portfolio variance): "
            f"single-stock volatility the six-factor model can't explain (earnings, industry news, etc.) — "
            f"usually reduced by increasing the number of holdings and diversifying across sectors.</span></div>"
        ),
    ]
    return "".join(rows)


def build_stock_loadings_narrative(holdings: list, factor_cols: list) -> str:
    """
    Generates the narrative HTML for the "factor loadings matrix" chart: for each factor, finds the
    holding with the highest and lowest exposure, to help spot "who's the most extreme on which
    dimension" without eyeballing a wall of grouped bars.
    holdings: [{'symbol': 'AAPL', 'MKT': 1.13, 'HML': -0.49, ...}, ...] (e.g. portfolio_df.to_dict('records')).
    """
    rows = []
    for col in factor_cols:
        if col not in STYLE_TAGS:
            continue
        valid = [(h.get('symbol'), h.get(col)) for h in holdings if h.get(col) is not None and not pd_isna(h.get(col))]
        if not valid:
            continue

        max_symbol, max_val = max(valid, key=lambda t: t[1])
        min_symbol, min_val = min(valid, key=lambda t: t[1])
        spread = max_val - min_val
        factor_name = _factor_display_name(col)
        color = _POS_COLOR if spread >= 0.3 else _NEUTRAL_COLOR

        rows.append(
            f"<div style='height:60px; box-sizing:border-box; display:flex; align-items:center; line-height:1.3; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {color};'>"
            f"<span><b>{factor_name}</b>: <b>{max_symbol}</b> highest (β={max_val:.2f}), "
            f"<b>{min_symbol}</b> lowest (β={min_val:.2f}) — spread of {spread:.2f} within the portfolio.</span></div>"
        )

    return "".join(rows)


def build_risk_contribution_narrative(contrib_rows: list) -> str:
    """
    Generates the narrative HTML for the "marginal risk contribution" chart: for each holding,
    compares its share of portfolio risk to its share of portfolio weight, and flags whether it's
    punching above or below its weight in driving total volatility.
    contrib_rows: [{'symbol': 'AAPL', 'weight_pct': 0.34, 'risk_pct': 0.51}, ...], any order.
    A "risk multiplier" of risk_pct / weight_pct >= 1.3 means that holding contributes
    disproportionately more risk than its capital allocation would suggest (concentration risk);
    <= 0.7 means it's a diversifying holding that contributes less risk than its weight.
    """
    rows = []
    for r in contrib_rows:
        symbol = r['symbol']
        weight_pct = r['weight_pct']
        risk_pct = r['risk_pct']
        multiplier = (risk_pct / weight_pct) if weight_pct > 0 else float('inf')

        if multiplier >= 1.3:
            color = _NEG_COLOR
            note = f"punches above its weight in driving portfolio risk ({multiplier:.1f}× its capital share)"
        elif multiplier <= 0.7:
            color = _POS_COLOR
            note = f"a diversifying holding — contributes less risk than its weight ({multiplier:.1f}× its capital share)"
        else:
            color = _NEUTRAL_COLOR
            note = "its risk contribution roughly tracks its portfolio weight"

        rows.append(
            f"<div style='height:56px; box-sizing:border-box; display:flex; align-items:center; "
            f"line-height:1.3; font-size:13.5px; padding-left:10px; border-left:3px solid {color};'>"
            f"<span><b>{symbol}</b>: {weight_pct:.1%} of capital, "
            f"<span style='color:{color}; font-weight:600;'>{risk_pct:.1%}</span> of portfolio risk — {note}.</span>"
            f"</div>"
        )

    return "".join(rows)


def build_var_narrative(var_res: dict) -> str:
    """
    Generates the narrative HTML for the "Value at Risk / Conditional VaR" card — explains what
    the two numbers mean in plain language, since VaR alone ("the loss won't exceed X most of the
    time") is easy to misread as "the loss can't exceed X".
    var_res: the dict returned by RiskDecompositionEngine.compute_parametric_var().
    """
    conf = var_res["confidence_level"]
    horizon = var_res["horizon_days"]
    var_pct = var_res["var_pct"]
    cvar_pct = var_res["cvar_pct"]
    tail_pct = 1 - conf
    horizon_label = "1 trading day" if horizon == 1 else f"{horizon} trading days"

    return (
        f"<div style='font-size:13.5px; line-height:1.6;'>"
        f"<div style='margin-bottom:8px;'>"
        f"At a <b>{conf:.0%}</b> confidence level, this portfolio is not expected to lose more than "
        f"<span style='color:{_NEG_COLOR}; font-weight:600;'>{var_pct:.1%}</span> of its value over the next {horizon_label} "
        f"— under the model's normal-distribution assumption, a loss this large or larger happens in only "
        f"about <b>{tail_pct:.0%}</b> of periods like this one."
        f"</div>"
        f"<div>"
        f"VaR says nothing about how bad that remaining {tail_pct:.0%} tail can get. Conditional VaR (Expected Shortfall) fills that gap: "
        f"<i>when</i> the loss does breach VaR, its expected size is "
        f"<span style='color:{_NEG_COLOR}; font-weight:600;'>{cvar_pct:.1%}</span> — always ≥ VaR, and the gap between the two "
        f"is a rough read on how fat the portfolio's tail risk is."
        f"</div>"
        f"</div>"
    )


def build_correlation_narrative(symbols: list, correlation_matrix: list) -> str:
    """
    Generates the narrative HTML for the "holdings correlation matrix" heatmap: calls out the
    most-correlated pair (the diversification is weakest between these two) and the
    least-correlated (or most negatively correlated) pair (the strongest diversifier in the
    portfolio), instead of leaving the reader to eyeball an N x N grid.
    symbols: ordered list of tickers matching the matrix's row/column order.
    correlation_matrix: NxN list of lists (or ndarray), as returned by
        RiskDecompositionEngine.compute_correlation_matrix()['correlation_matrix'].
    """
    n = len(symbols)
    if n < 2:
        return "<div style='font-size:13.5px;'>Need at least two holdings to compute pairwise correlation.</div>"

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((symbols[i], symbols[j], correlation_matrix[i][j]))

    most_corr = max(pairs, key=lambda p: p[2])
    least_corr = min(pairs, key=lambda p: p[2])
    avg_corr = sum(p[2] for p in pairs) / len(pairs)

    def _corr_label(v: float) -> str:
        if v >= 0.7:
            return "very highly correlated"
        if v >= 0.4:
            return "moderately correlated"
        if v >= 0.0:
            return "weakly correlated"
        return "negatively correlated (a natural hedge)"

    rows = [
        (
            f"<div style='min-height:110px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_NEG_COLOR};'>"
            f"<span><b>Most correlated pair:</b> <b>{most_corr[0]}</b> and <b>{most_corr[1]}</b> "
            f"(ρ=<span style='color:{_NEG_COLOR}; font-weight:600;'>{most_corr[2]:.2f}</span>) — "
            f"{_corr_label(most_corr[2])}. Holding both adds less diversification than their separate "
            f"weights suggest; a shock hitting one tends to hit the other too.</span></div>"
        ),
        (
            f"<div style='min-height:110px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_POS_COLOR};'>"
            f"<span><b>Least correlated pair:</b> <b>{least_corr[0]}</b> and <b>{least_corr[1]}</b> "
            f"(ρ=<span style='color:{_POS_COLOR}; font-weight:600;'>{least_corr[2]:.2f}</span>) — "
            f"{_corr_label(least_corr[2])}. This is the strongest diversifying pair currently in the portfolio.</span></div>"
        ),
        (
            f"<div style='min-height:80px; display:flex; align-items:center; font-size:13.5px;"
            f"padding-left:10px; border-left:3px solid {_NEUTRAL_COLOR};'>"
            f"<span>Average pairwise correlation across all {len(pairs)} holding pairs: "
            f"<b>{avg_corr:.2f}</b>. The closer this sits to 1, the less benefit additional holdings "
            f"are providing — risk starts behaving like one big correlated bet rather than a diversified basket.</span></div>"
        ),
    ]
    return "".join(rows)


def render_streamlit(lines: list[DiagnosisLine], container=None):
    """Renders a list of DiagnosisLine objects as Streamlit components.
    Pass `st` or `st.container()` as `container` to embed it into an existing layout.
    """
    import streamlit as st
    target = container or st

    level_to_fn = {
        "good": target.success,
        "info": target.info,
        "warning": target.warning,
        "danger": target.error,
    }

    for line in lines:
        fn = level_to_fn.get(line.level, target.info)
        fn(f"{line.icon} **{line.headline}**\n\n{line.detail}")


# Sector tag colors: hashed into a fixed palette so the same sector always gets the same color app-wide
_SECTOR_PALETTE = [
    "#2563eb", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#dc2626", "#4d7c0f",
]


def sector_color(sector: str) -> str:
    """Generates a stable accent color for a GICS sector, used for the sector dot in the holdings panel etc.
    Uses zlib.crc32 rather than the builtin hash() because Python's string hash() is randomized per
    process (PYTHONHASHSEED) — the same sector would get a different color on every app restart.
    crc32 keeps it stable across process runs.
    """
    if not sector:
        return "#94a3b8"
    import zlib
    return _SECTOR_PALETTE[zlib.crc32(sector.encode()) % len(_SECTOR_PALETTE)]
