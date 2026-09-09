import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.db_client import RiskDatabase
from src.risk_decomposition import RiskDecompositionEngine
from src.risk_narrator import (
    diagnose_active_risk, diagnose_absolute_risk, render_streamlit,
    build_style_tags, build_factor_narrative, factor_bar_colors,
    build_portfolio_exposure_summary, build_active_tilt_summary,
    build_active_tilt_narrative, build_te_pie_narrative, active_tilt_bar_colors,
    sector_color, build_abs_variance_pie_narrative, build_stock_loadings_narrative,
    build_risk_contribution_narrative, build_var_narrative, build_correlation_narrative
)

# Base page config: the nav lives in the native left sidebar, which has its own expand/collapse control
st.set_page_config(
    page_title="AlphaRisk Quantitative Portfolio Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom metric-card styling + left nav styling + a compact type scale to match the tighter chart heights
st.markdown("""
    <style>
    /* Streamlit's wide layout reserves a large top gap for the fixed header toolbar (Deploy/menu).
       Trim it, but not below the toolbar's own height — 1.5rem was too aggressive and let content
       render underneath the toolbar, clipping it. */
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: 3.5rem !important; }

    .stMetric { background-color: #f8fafc; padding: 8px 14px; border-radius: 8px; border: 1px solid #e2e8f0; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }

    /* Compact type scale: headings, body text, captions, and alert boxes all shrink together.
       Not scoped to a ".main" ancestor — recent Streamlit versions don't use that class on the
       content container, so a scoped selector silently matches nothing. */
    h1 { font-size: 1.5rem !important; margin: 0.4rem 0 !important; }
    h2 { font-size: 1.25rem !important; margin: 0.4rem 0 !important; }
    h3 { font-size: 1.05rem !important; margin: 0.5rem 0 0.2rem 0 !important; }
    h4 { font-size: 0.95rem !important; margin: 0.4rem 0 0.2rem 0 !important; }
    h5 { font-size: 0.88rem !important; margin: 0.3rem 0 0.2rem 0 !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p { font-size: 0.85rem !important; }
    [data-testid="stCaptionContainer"] { font-size: 0.75rem !important; }
    [data-testid="stAlert"] { font-size: 0.82rem !important; padding: 0.55rem 0.8rem !important; }
    [data-testid="stAlert"] p { font-size: 0.82rem !important; }
    .stButton button { font-size: 0.85rem !important; }
    [data-testid="stExpander"] summary { font-size: 0.85rem !important; }
    .stDataFrame, .stDataFrame div { font-size: 0.82rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding-top: 0.4rem !important; padding-bottom: 0.4rem !important; }

    /* Left nav: make the buttons read as nav items, not form buttons */
    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stButton button {
        justify-content: flex-start;
        text-align: left;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 0.9rem;
        border: 1px solid transparent;
        transition: background-color 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background-color: transparent;
        color: #334155;
    }
    [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        background-color: #e2e8f0;
        color: #0f172a;
        border-color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    db = RiskDatabase("factor_risk.db")
    exposures_df = db.load_exposures_wide()
    sec_df = pd.read_sql_query("SELECT symbol, sec_name, sector FROM securities", db.conn)
    factors_df = db.load_factors_wide()

    if not exposures_df.empty and not sec_df.empty:
        exposures_df = pd.merge(exposures_df, sec_df, on="symbol", how="left")
    return exposures_df, factors_df

@st.cache_data
def build_symbol_label_map(df: pd.DataFrame) -> dict:
    """symbol -> "AAPL · Apple Inc. · Information Technology", used for fuzzy search in the dialog
    & holding cards. Cached and vectorized (no .iterrows()) since this now runs over the full
    ~500-ticker universe rather than a small subset.
    """
    unique_df = df.drop_duplicates(subset=['symbol'])
    sec_name = unique_df['sec_name'].fillna('Unknown company')
    sector = unique_df['sector'].fillna('Unknown sector')
    labels = unique_df['symbol'] + ' · ' + sec_name + ' · ' + sector
    return dict(zip(unique_df['symbol'], labels))

exposures_df, factors_df = load_data()

if exposures_df.empty:
    st.error("Database is empty — please run `python main.py` in a terminal first.")
    st.stop()

# Make sure SPY exists in the database as the benchmark anchor
spy_record = exposures_df[exposures_df['symbol'] == 'SPY']
if spy_record.empty:
    st.error("SPY (S&P 500) benchmark data not found in the database — make sure main.py includes SPY and has computed its exposures.")
    st.stop()

meta_cols = {'symbol', 'as_of_date', 'idiosyncratic_var', 'r_squared', 'sec_name', 'sector'}
factor_cols = [c for c in exposures_df.columns if c not in meta_cols]

# ----------------- Asset universe prep -----------------
non_bench_df = exposures_df[exposures_df['symbol'] != 'SPY']
all_symbols = sorted(non_bench_df['symbol'].unique().tolist())
all_sectors = sorted(non_bench_df['sector'].dropna().unique().tolist())

_symbol_label_map = build_symbol_label_map(non_bench_df)

SELECTED_SYMBOLS_KEY = "selected_symbols"
if SELECTED_SYMBOLS_KEY not in st.session_state:
    default_picks = [s for s in ['AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM'] if s in all_symbols]
    if not default_picks and len(all_symbols) >= 2:
        default_picks = all_symbols[:2]
    st.session_state[SELECTED_SYMBOLS_KEY] = default_picks

# Four sub-pages: holdings management / active risk / absolute risk / stock detail, switched via the left nav
NAV_PAGES = [
    ("holdings", "📌", "Holdings"),
    ("active", "📐", "Active Risk"),
    ("absolute", "📊", "Absolute Risk"),
    ("detail", "🔍", "Stock Detail"),
]
CURRENT_PAGE_KEY = "current_page"
if CURRENT_PAGE_KEY not in st.session_state:
    st.session_state[CURRENT_PAGE_KEY] = "active"


# ----------------- Dedicated search dialog: TradingView-style "+ Add Stock" -----------------
@st.dialog("Search & Add Stocks", width="large")
def open_symbol_search_dialog():
    search_tab, sector_tab = st.tabs(["🔍 Search by Ticker / Name", "🏷️ Bulk-Add by Sector"])

    with search_tab:
        query = st.text_input("Search by ticker, company name, or sector keyword:", key="symbol_search_query")
        candidates = all_symbols
        if query.strip():
            q = query.strip().lower()
            candidates = [s for s in all_symbols if q in _symbol_label_map[s].lower()]

        st.caption(f"{len(candidates)} matches" if query.strip() else f"Full universe, {len(candidates)} tickers (type a keyword to filter)")

        for s in candidates[:80]:
            row_cols = st.columns([5, 1])
            row_cols[0].markdown(_symbol_label_map[s])
            if s in st.session_state[SELECTED_SYMBOLS_KEY]:
                row_cols[1].markdown("✅ Held")
            else:
                if row_cols[1].button("Add", key=f"dlg_add_{s}"):
                    st.session_state[SELECTED_SYMBOLS_KEY] = st.session_state[SELECTED_SYMBOLS_KEY] + [s]
                    st.rerun()

        if len(candidates) > 80:
            st.caption(f"Showing the first 80 results — narrow your search to see more ({len(candidates)} total matches).")

    with sector_tab:
        quick_sector = st.selectbox("Choose a sector:", options=all_sectors, key="quick_add_sector")
        sector_symbols = sorted(non_bench_df.loc[non_bench_df['sector'] == quick_sector, 'symbol'].unique().tolist())
        st.caption(f"\"{quick_sector}\" has {len(sector_symbols)} constituents.")
        if st.button("Add every stock in this sector to holdings", key="dlg_add_sector", width="stretch"):
            st.session_state[SELECTED_SYMBOLS_KEY] = sorted(set(st.session_state[SELECTED_SYMBOLS_KEY]) | set(sector_symbols))
            st.rerun()


selected_symbols = st.session_state[SELECTED_SYMBOLS_KEY]
# Reindex to match selected_symbols' order exactly (not whichever order the rows happen to sit in
# exposures_df) — w_p_active is built by iterating selected_symbols, and decompose_risk() /
# compute_risk_contributions() zip weights against these rows positionally with no symbol-based
# re-matching, so a mismatched row order here silently pairs each weight with the wrong stock.
portfolio_df = exposures_df.drop_duplicates(subset=['symbol']).set_index('symbol').loc[selected_symbols].reset_index()

# With no holdings, only the "Holdings" page is usable — the other two pages need at least 1 stock to compute risk
current_page = st.session_state[CURRENT_PAGE_KEY]
if not selected_symbols and current_page != "holdings":
    current_page = "holdings"
    st.session_state[CURRENT_PAGE_KEY] = "holdings"

# ----------------- Portfolio weights: the input boxes live on the "Holdings" page; compute from persisted state here -----------------
# Weights are always "manually typed share + auto-normalized to 100%" — there's no separate equal-weight
# toggle. A newly added stock defaults to 1/N (stored as a percentage), and the user can type a number
# on its holding card at any time to adjust it.
if selected_symbols:
    raw_weights = {}
    for s in selected_symbols:
        weight_key = f"weight_pct_{s}"
        if weight_key not in st.session_state:
            st.session_state[weight_key] = round(100.0 / len(selected_symbols), 1)
        raw_weights[s] = st.session_state[weight_key] / 100.0
    w_vals = np.array([raw_weights[s] for s in selected_symbols])
    sum_w = np.sum(w_vals)
    w_p_active = w_vals / sum_w if sum_w > 0 else np.ones(len(selected_symbols)) / len(selected_symbols)

    # ----------------- Core linear-algebra computation -----------------
    calc_df = pd.concat([portfolio_df, spy_record]).drop_duplicates(subset=['symbol']).reset_index(drop=True)
    n_assets = len(calc_df)

    B_augmented = calc_df[factor_cols].values
    specific_vars_augmented = calc_df['idiosyncratic_var'].values

    if not factors_df.empty and len(factors_df) > 10:
        factors_clean = factors_df[[c for c in factor_cols if c in factors_df.columns]]
        factor_cov = np.cov(factors_clean.values, rowvar=False)
    else:
        factor_cov = np.eye(len(factor_cols)) * 0.0001

    # Absolute risk
    B_port = portfolio_df[factor_cols].values
    specific_vars_port = portfolio_df['idiosyncratic_var'].values
    abs_res = RiskDecompositionEngine.decompose_risk(w_p_active, B_port, factor_cov, specific_vars_port, factor_cols)
    rc_res = RiskDecompositionEngine.compute_risk_contributions(
        w_p_active, B_port, factor_cov, specific_vars_port, list(portfolio_df['symbol'])
    )

    # Build the active-risk weight vectors
    w_port_full = np.zeros(n_assets)
    for idx, s in enumerate(calc_df['symbol']):
        if s in selected_symbols:
            s_idx = selected_symbols.index(s)
            w_port_full[idx] = w_p_active[s_idx]

    w_bench_full = np.zeros(n_assets)
    spy_idx = calc_df['symbol'].tolist().index('SPY')
    w_bench_full[spy_idx] = 1.0

    act_res = RiskDecompositionEngine.decompose_active_risk(
        w_port_full,
        w_bench_full,
        B_augmented,
        factor_cov,
        specific_vars_augmented,
        factor_cols
    )
else:
    w_p_active = None
    abs_res = None
    rc_res = None
    act_res = None


def render_detail_page():
    """Constituent exposure table + full per-stock factor-exposure narrative."""
    st.markdown("## 🔍 Stock Detail")
    st.caption("Constituent-level exposures, active deviation vs. SPY, and the full six-factor narrative for any single holding.")

    st.markdown(
        "**Constituent Exposures, Active Deviation & Difference vs. Benchmark (SPY)**",
        help="Column notes:\n• w_p: current portfolio weight\n• w_b: benchmark (SPY) weight (100%)\n• Δw: active over/underweight (w_p - w_b)\n• Style Tag: an auto-generated style profile based on the six-factor exposure (pick a stock below for the full narrative)\n• R²: the goodness-of-fit of the six-factor model for this stock's daily returns"
    )
    st.caption("The table has been trimmed to its core columns to avoid horizontal scrolling; expand a single stock below for its full factor exposure.")

    display_df = calc_df.copy()
    display_df['Portfolio_w'] = w_port_full
    display_df['Benchmark_w'] = w_bench_full
    display_df['Active_w'] = act_res['delta_w']
    display_df['style_tag'] = display_df.apply(
        lambda r: build_style_tags(r.to_dict(), factor_cols), axis=1
    )

    col_config = {
        "symbol": "Ticker",
        "sec_name": "Company",
        "sector": "Sector",
        "Portfolio_w": st.column_config.ProgressColumn("Portfolio Wt. (w_p)", min_value=0.0, max_value=1.0, format="%.1f%%"),
        "Benchmark_w": st.column_config.ProgressColumn("Benchmark Wt. (w_b)", min_value=0.0, max_value=1.0, format="%.1f%%"),
        "Active_w": st.column_config.NumberColumn("Active Wt. (Δw)", format="%+.1f%%"),
        "style_tag": st.column_config.TextColumn("Style Tag", width="medium"),
        "r_squared": st.column_config.NumberColumn("Fit (R²)", format="%.1%"),
    }

    ordered_cols = ['symbol', 'sec_name', 'sector', 'Portfolio_w', 'Benchmark_w', 'Active_w', 'style_tag', 'r_squared']

    st.dataframe(
        display_df[ordered_cols],
        column_config=col_config,
        width="stretch",
        hide_index=True
    )

    st.markdown("**🔍 Full Per-Stock Factor Exposure**")
    detail_options = display_df['symbol'].tolist()
    # A previously selected symbol may have been removed from holdings on the "Holdings" page since
    # the last visit here — reset the stale session-state value before the widget is created, or
    # Streamlit raises an error because the stored value is no longer a valid option.
    if st.session_state.get("detail_symbol_selector") not in detail_options:
        st.session_state["detail_symbol_selector"] = detail_options[0]

    detail_symbol = st.selectbox(
        "Pick a stock to see its full six-factor exposure and narrative:",
        options=detail_options,
        key="detail_symbol_selector"
    )

    detail_row = display_df[display_df['symbol'] == detail_symbol].iloc[0].to_dict()

    st.markdown(f"##### {detail_symbol} | {detail_row.get('sec_name', '')}")
    st.caption(f"Sector: {detail_row.get('sector', 'Unknown')} | Fit R² = {detail_row.get('r_squared', 0):.1%} "
               f"| Idiosyncratic Variance = {detail_row.get('idiosyncratic_var', 0):.2e}")

    detail_left, detail_right = st.columns([1, 1])

    with detail_left:
        detail_colors = factor_bar_colors(detail_row, factor_cols)
        detail_values = [detail_row.get(c) for c in factor_cols]
        detail_fig = go.Figure()
        detail_fig.add_trace(go.Bar(
            x=detail_values,
            y=factor_cols,
            orientation='h',
            marker_color=detail_colors,
            text=[f"{v:.2f}" for v in detail_values],
            textposition='outside'
        ))
        detail_fig.update_layout(
            height=380,
            margin=dict(l=10, r=40, t=10, b=10),
            template='plotly_white',
            xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748b'),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(detail_fig, width="stretch")
        st.caption("Color intensity reflects exposure direction: blue = positive exposure, red = negative exposure, gray = near-neutral.")

    with detail_right:
        st.markdown(build_factor_narrative(detail_symbol, detail_row, factor_cols), unsafe_allow_html=True)


def render_holdings_page():
    st.markdown("## 📌 Holdings")
    st.caption("Search for, add, and remove constituent stocks, and type a share directly on each card to adjust its portfolio weight (auto-normalized to 100%; the benchmark is fixed at 100% SPY).")

    if st.button("＋ Add Stock", key="open_search_dialog", type="primary"):
        open_symbol_search_dialog()

    if not selected_symbols:
        st.info("No holdings yet — click the button above to search for and add a stock (at least 1).")
        return

    st.markdown(f"#### Current Holdings ({len(selected_symbols)})")
    st.caption("Totals 100% — adjust a single weight below and the rest re-normalize proportionally")

    n_cols = 3
    card_cols = st.columns(n_cols)
    for i, s in enumerate(selected_symbols):
        sec_row = portfolio_df[portfolio_df['symbol'] == s].iloc[0]
        sector = sec_row.get('sector') or 'Unknown sector'
        dot_color = sector_color(sector)
        weight_pct = w_p_active[i]

        with card_cols[i % n_cols]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex; align-items:baseline; gap:8px;'>"
                    f"<b style='font-size:14px;'>{s}</b>"
                    f"<span style='background:#eff6ff; color:#2563eb; font-size:11px; "
                    f"font-weight:600; padding:1px 7px; border-radius:10px;'>{weight_pct:.1%}</span>"
                    f"</div>"
                    f"<div style='color:#64748b; font-size:11px; margin-top:2px;'>"
                    f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; "
                    f"background:{dot_color}; margin-right:5px;'></span>"
                    f"{sec_row.get('sec_name', '')} · {sector}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.number_input(
                    "Weight (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
                    key=f"weight_pct_{s}", label_visibility="collapsed"
                )
                if st.button("Remove", key=f"remove_{s}", width="stretch"):
                    st.session_state[SELECTED_SYMBOLS_KEY] = [x for x in st.session_state[SELECTED_SYMBOLS_KEY] if x != s]
                    st.rerun()


def render_active_risk_page():
    st.info(f"📌 {build_active_tilt_summary(act_res['active_tilts'], factor_cols)}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Annualized Tracking Error",
        f"{act_res['tracking_error']:.2%}",
        help="Formula:\nTE = √[ 252 × (Δwᵀ B Σ_F Bᵀ Δw + Δwᵀ D Δw) ]\n\nMeaning:\nThe annualized standard deviation of the portfolio's active return relative to the S&P 500 (SPY). The larger the TE, the more sharply the portfolio's return diverges from the benchmark."
    )
    k2.metric(
        "Active Factor Risk Share",
        f"{act_res['active_factor_ratio']:.1%}",
        delta="Macro & Style Tilts",
        help="Formula:\nActive Factor Ratio = (Δwᵀ B Σ_F Bᵀ Δw) / TE²\n\nMeaning:\nThe share of tracking-error variance contributed by macro/style factor misalignment (e.g. overweighting momentum or underweighting value). Can be reduced through style-neutralizing hedges."
    )
    k3.metric(
        "Active Specific Risk Share",
        f"{act_res['active_specific_ratio']:.1%}",
        delta="Stock Picking Alpha",
        delta_color="inverse",
        help="Formula:\nActive Specific Ratio = (Δwᵀ D Δw) / TE²\n\nMeaning:\nThe share of tracking-error variance contributed by pure stock-picking concentration. The more diversified the holdings, the closer this ratio gets to 0%."
    )
    k4.metric(
        "Portfolio Absolute Annualized Volatility",
        f"{abs_res['annualized_vol']:.2%}",
        help="Formula:\nAnn. Vol = √(252 × σ_p²)\n\nMeaning:\nThe annualized standard deviation of the portfolio's own absolute daily returns (unhedged against the benchmark)."
    )

    diagnosis_lines = diagnose_active_risk(act_res, factor_cols, len(selected_symbols))
    with st.expander(f"📋 Portfolio Diagnostic Summary ({len(diagnosis_lines)} insights)", expanded=False):
        st.caption("Generated automatically by a rule engine from the current holdings, to help you quickly understand what the numbers above mean.")
        render_streamlit(diagnosis_lines)

    tilt_series = act_res['active_tilts']

    st.markdown(
        "**Active Factor Tilts (Δβ)**",
        help="Formula:\nΔβ = Bᵀ × Δw = Bᵀ × (w_portfolio - w_benchmark)\n\nFactor cheat sheet:\n• MKT: overall market beta\n• SMB: size (small - large)\n• HML: book-to-market (value - growth)\n• RMW: profitability (robust - weak)\n• CMA: investment style (conservative - aggressive)\n• MOM: momentum (winners - losers)\n\nBars pointing up (blue) = overweight, bars pointing down (red) = underweight."
    )
    st.caption("Bars show the portfolio's over- (blue, up) or under- (red, down) weight relative to SPY on each style factor; taller bars mean a more extreme bet on that factor.")

    tilt_left, tilt_right = st.columns([1, 1])
    with tilt_left:
        tilt_fig = go.Figure()
        bar_colors = active_tilt_bar_colors(tilt_series, factor_cols)
        tilt_fig.add_trace(go.Bar(
            x=list(tilt_series.values()),
            y=list(tilt_series.keys()),
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in tilt_series.values()],
            textposition='outside'
        ))
        tilt_fig.update_layout(
            height=380,
            margin=dict(l=10, r=40, t=20, b=10),
            template='plotly_white',
            xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748b'),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(tilt_fig, width="stretch")

    with tilt_right:
        st.markdown(build_active_tilt_narrative(tilt_series, factor_cols), unsafe_allow_html=True)

    st.markdown(
        "**Tracking-Error Variance Decomposition**",
        help="Quadratic-form decomposition:\nTE² = [Δwᵀ (B Σ_F Bᵀ) Δw] + [Δwᵀ D Δw]\n\n• Blue: active factor risk (driven by style deviation)\n• Orange: active specific risk (driven by stock-picking concentration)"
    )
    st.caption("Splits the tracking error above into two pieces: blue is the part driven by \"style bets\" (can be reduced via neutralizing hedges), orange is the part driven by \"stock-picking concentration\" (can be reduced by diversifying holdings).")

    pie_left, pie_right = st.columns([1, 1])
    with pie_left:
        pie_fig = go.Figure(data=[go.Pie(
            labels=['Active Factor Risk', 'Active Specific Risk'],
            values=[act_res['active_factor_var'], act_res['active_specific_var']],
            hole=0.6,
            textinfo='none',
            marker=dict(colors=['#2563eb', '#f97316'])
        )])
        pie_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            template='plotly_white',
            annotations=[dict(
                text=f"<b>{act_res['tracking_error']:.2%}</b><br><span style='font-size:12px;color:#64748b'>Ann. TE</span>",
                x=0.5, y=0.5, font_size=15, showarrow=False
            )]
        )
        st.plotly_chart(pie_fig, width="stretch")

    with pie_right:
        st.markdown(build_te_pie_narrative(act_res), unsafe_allow_html=True)


def render_absolute_risk_page():
    st.info(f"📌 {build_portfolio_exposure_summary(abs_res['portfolio_exposure'], factor_cols)}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Portfolio Annualized Total Volatility",
        f"{abs_res['annualized_vol']:.2%}",
        help="Formula:\nσ_ann = √[ 252 × (wᵀ B Σ_F Bᵀ w + wᵀ D w) ]\n\nMeaning:\nThe annualized standard deviation of the holdings' overall risk exposure, unhedged against any benchmark."
    )
    k2.metric(
        "Systematic Factor Risk Share",
        f"{abs_res['factor_var_ratio']:.1%}",
        delta=f"{len(factor_cols)}-Factor Model",
        help="Formula:\nFactor Var Ratio = (wᵀ B Σ_F Bᵀ w) / σ_p²\n\nMeaning:\nThe share of total portfolio volatility that can be jointly explained by the Carhart six factors."
    )
    k3.metric(
        "Idiosyncratic Risk Share",
        f"{abs_res['specific_var_ratio']:.1%}",
        delta="Idiosyncratic Risk",
        delta_color="inverse",
        help="Formula:\nSpecific Var Ratio = (wᵀ D w) / σ_p²\n\nMeaning:\nThe share of residual, single-stock volatility that the multi-factor model can't explain (single-stock tail risk)."
    )
    k4.metric(
        "Active Holdings Count",
        f"{len(selected_symbols)} tickers",
        help="The number of active stocks currently included in the portfolio's risk calculation."
    )

    diagnosis_lines = diagnose_absolute_risk(abs_res, factor_cols, len(selected_symbols))
    with st.expander(f"📋 Portfolio Diagnostic Summary ({len(diagnosis_lines)} insights)", expanded=False):
        st.caption("Generated automatically by a rule engine from the current holdings, to help you quickly understand what the numbers above mean.")
        render_streamlit(diagnosis_lines)

    # KPI cards + diagnostic summary above stay always visible as the page header; everything
    # else is grouped into tabs so the page isn't one long scroll of 6 stacked chart blocks.
    tab_exposure, tab_decomp, tab_corr = st.tabs(
        ["⚖️ Factor Exposure", "🧮 Risk Decomposition", "🔗 Correlation"]
    )

    with tab_exposure:
        # --- Portfolio-level six-factor exposure: bar chart + per-factor narrative ---
        st.markdown(
            "**Portfolio-Level Factor Exposure (B_p = Bᵀw)**",
            help="Formula:\nB_p = Bᵀ × w\n\nWeights each stock's factor loadings by its portfolio weight and sums them, giving the six-factor profile of the \"portfolio as a whole\" rather than stock-by-stock exposures."
        )
        st.caption("B_p has the same shape as a single stock's β and reads the same way: judge MKT by its deviation from 1, and every other factor by its deviation from 0; the narrative colors on the right match the bar directions.")

        exposure_series = abs_res['portfolio_exposure']
        exposure_left, exposure_right = st.columns([1, 1])
        with exposure_left:
            exposure_fig = go.Figure()
            exposure_fig.add_trace(go.Bar(
                x=list(exposure_series.values()),
                y=list(exposure_series.keys()),
                orientation='h',
                marker_color=factor_bar_colors(exposure_series, factor_cols),
                text=[f"{v:+.2f}" for v in exposure_series.values()],
                textposition='outside'
            ))
            exposure_fig.update_layout(
                height=380,
                margin=dict(l=10, r=40, t=20, b=10),
                template='plotly_white',
                xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748b'),
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(exposure_fig, width="stretch")

        with exposure_right:
            st.markdown(
                build_factor_narrative("Portfolio", exposure_series, factor_cols, subject_label="this portfolio"),
                unsafe_allow_html=True
            )

        # --- Factor loadings matrix: grouped bar chart + per-factor highest/lowest narrative ---
        st.markdown(
            "**Factor Loadings Matrix (B)**",
            help="Regression model:\nR_i - R_f = α_i + Σ β_{i,k} F_k + ε_i\n\nShows each holding's multi-factor OLS regression slope β across the six systematic style factors — compare it to the portfolio-level exposure above to see which stocks are driving the overall style."
        )
        st.caption("The panel on the right calls out, per factor, the holding with the highest/lowest exposure: blue means the stocks diverge noticeably on that factor (spread ≥ 0.3), gray means they're fairly aligned on that dimension.")

        matrix_left, matrix_right = st.columns([3, 2])
        with matrix_left:
            bar_fig = go.Figure()
            palette = ['#2563eb', '#64748b', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2']
            for idx, col in enumerate(factor_cols):
                bar_fig.add_trace(go.Bar(
                    x=portfolio_df['symbol'],
                    y=portfolio_df[col],
                    name=col,
                    marker_color=palette[idx % len(palette)]
                ))
            bar_fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                barmode='group',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(bar_fig, width="stretch")

        with matrix_right:
            holdings_records = portfolio_df[['symbol'] + factor_cols].to_dict('records')
            st.markdown(build_stock_loadings_narrative(holdings_records, factor_cols), unsafe_allow_html=True)

    with tab_decomp:
        # --- Total variance decomposition: pie chart + two-slice narrative ---
        st.markdown(
            "**Total Variance Decomposition**",
            help="The cornerstone of a multi-factor risk model:\nσ_p² = [wᵀ (B Σ_F Bᵀ) w] + [wᵀ D w]\n\nCleanly splits the portfolio's total variance into: systematic macro factor variance vs. idiosyncratic single-stock variance."
        )
        st.caption("Splits the annualized volatility above into two pieces: blue is the systematic part the six factors can explain, orange is the idiosyncratic part they can't.")

        abs_pie_left, abs_pie_right = st.columns([1, 1])
        with abs_pie_left:
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Systematic Factor Risk', 'Idiosyncratic Risk'],
                values=[abs_res['factor_variance'], abs_res['specific_variance']],
                hole=0.6,
                textinfo='none',
                marker=dict(colors=['#2563eb', '#f97316'])
            )])
            pie_fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                template='plotly_white',
                annotations=[dict(
                    text=f"<b>{abs_res['factor_var_ratio']:.1%}</b><br><span style='font-size:12px;color:#64748b'>Factor Risk</span>",
                    x=0.5, y=0.5, font_size=15, showarrow=False
                )]
            )
            st.plotly_chart(pie_fig, width="stretch")

        with abs_pie_right:
            st.markdown(build_abs_variance_pie_narrative(abs_res), unsafe_allow_html=True)

        # --- Marginal risk contribution: which holdings actually drive portfolio risk ---
        st.markdown(
            "**Marginal Risk Contribution**",
            help="Euler decomposition of volatility:\nMCTR_i = (Σw)_i / σ_p ,  RC_i = w_i × MCTR_i ,  Σ RC_i = σ_p\n\nSplits total portfolio volatility additively across holdings using the full covariance matrix (factor + idiosyncratic) — not just each stock's standalone factor loadings. A holding can have modest weight but still drive most of the risk if it's highly correlated with the rest of the portfolio."
        )
        st.caption("Compares each holding's share of portfolio risk to its share of capital: contributing much more risk than its weight flags a concentration risk; contributing less means it's diversifying the portfolio.")

        contrib_symbols = list(portfolio_df['symbol'])
        contrib_rows = sorted(
            (
                {
                    "symbol": s,
                    "weight_pct": w_p_active[i],
                    "risk_pct": rc_res["risk_contribution_pct"][s],
                }
                for i, s in enumerate(contrib_symbols)
            ),
            key=lambda r: r["risk_pct"],
            reverse=True
        )

        contrib_left, contrib_right = st.columns([1, 1])
        with contrib_left:
            contrib_fig = go.Figure()
            contrib_fig.add_trace(go.Bar(
                x=[r["risk_pct"] for r in contrib_rows],
                y=[r["symbol"] for r in contrib_rows],
                orientation='h',
                marker_color='#2563eb',
                text=[f"{r['risk_pct']:.1%}" for r in contrib_rows],
                textposition='outside'
            ))
            # 'outside' text labels sit past the bar tip, but autorange only fits the bar values
            # themselves — with no headroom, the longest label (the top bar) gets clipped by the
            # plot area. Pad the axis max by 20% (and cover the zero-or-empty case) so every label
            # has room to render.
            max_risk_pct = max((r["risk_pct"] for r in contrib_rows), default=0.0)
            axis_max = max_risk_pct * 1.2 if max_risk_pct > 0 else 0.1
            contrib_fig.update_layout(
                height=max(240, 56 * len(contrib_rows) + 40),
                margin=dict(l=10, r=40, t=20, b=10),
                template='plotly_white',
                bargap=0.35,
                xaxis=dict(title="Share of Total Portfolio Risk", tickformat='.0%', range=[0, axis_max], zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748b'),
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(contrib_fig, width="stretch")

        with contrib_right:
            st.markdown(build_risk_contribution_narrative(contrib_rows), unsafe_allow_html=True)

        # --- Value at Risk / Conditional VaR: parametric (Gaussian), reuses the same daily variance ---
        st.markdown(
            "**Value at Risk (VaR) & Conditional VaR**",
            help="Parametric (Gaussian) VaR/CVaR:\nVaR = z × σ_daily × √horizon ,  CVaR = σ_daily × √horizon × φ(z) / (1-confidence)\n\nReuses the same portfolio daily variance as the decomposition above. Assumes normally distributed daily returns with zero drift — a simplification appropriate for short horizons, not a substitute for historical/Monte Carlo VaR on fat-tailed portfolios."
        )
        st.caption("VaR: the loss this portfolio is not expected to exceed, at the chosen confidence level and horizon. CVaR (Expected Shortfall): the average loss in the tail scenarios where VaR is breached — always ≥ VaR.")

        var_conf_col, var_horizon_col = st.columns([1, 1])
        with var_conf_col:
            var_confidence = st.selectbox(
                "Confidence level", options=[0.95, 0.99], format_func=lambda v: f"{v:.0%}", key="var_confidence_level"
            )
        with var_horizon_col:
            var_horizon = st.selectbox(
                "Horizon", options=[1, 5, 10], format_func=lambda d: f"{d} trading day" + ("s" if d > 1 else ""), key="var_horizon_days"
            )

        var_res = RiskDecompositionEngine.compute_parametric_var(
            abs_res["total_variance"], confidence_level=var_confidence, horizon_days=var_horizon
        )

        var_m1, var_m2 = st.columns(2)
        var_m1.metric(f"VaR ({var_confidence:.0%}, {var_horizon}d)", f"{var_res['var_pct']:.1%}")
        var_m2.metric(f"CVaR / Expected Shortfall ({var_confidence:.0%}, {var_horizon}d)", f"{var_res['cvar_pct']:.1%}")

        var_chart_left, var_chart_right = st.columns([1, 1])
        with var_chart_left:
            var_fig = go.Figure()
            var_fig.add_trace(go.Bar(
                x=[var_res["var_pct"], var_res["cvar_pct"]],
                y=["VaR", "CVaR"],
                orientation='h',
                marker_color=['#f59e0b', '#ef4444'],
                text=[f"{var_res['var_pct']:.1%}", f"{var_res['cvar_pct']:.1%}"],
                textposition='outside'
            ))
            # Same 'outside'-label headroom fix as the Marginal Risk Contribution chart above —
            # otherwise the CVaR bar's label gets clipped since it's always the larger of the two.
            var_axis_max = var_res["cvar_pct"] * 1.3 if var_res["cvar_pct"] > 0 else 0.1
            var_fig.update_layout(
                height=180,
                margin=dict(l=10, r=40, t=10, b=10),
                template='plotly_white',
                bargap=0.5,
                xaxis=dict(
                    title=f"Potential Loss ({var_horizon}d, {var_confidence:.0%})",
                    tickformat='.1%', range=[0, var_axis_max],
                    zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748b'
                ),
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(var_fig, width="stretch")

        with var_chart_right:
            st.markdown(build_var_narrative(var_res), unsafe_allow_html=True)

    with tab_corr:
        # --- Holdings correlation matrix: same full covariance matrix as Marginal Risk Contribution,
        # normalized to [-1, 1] instead of left in variance units ---
        st.markdown(
            "**Holdings Correlation Matrix**",
            help="Formula:\nΣ = B Σ_F Bᵀ + D ,  Corr_ij = Σ_ij / (σ_i σ_j)\n\nPairwise correlation implied by the full covariance matrix, not just shared factor betas — two stocks with identical factor loadings can still show up weakly correlated here if idiosyncratic variance dominates."
        )
        st.caption("Red = holdings that tend to move together (weaker diversification between them); blue = holdings that offset each other (stronger diversification).")

        corr_res = RiskDecompositionEngine.compute_correlation_matrix(
            B_port, factor_cov, specific_vars_port, list(portfolio_df['symbol'])
        )
        corr_matrix = corr_res["correlation_matrix"]
        corr_symbols = corr_res["symbols"]

        corr_left, corr_right = st.columns([1, 1])
        with corr_left:
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_symbols,
                y=corr_symbols,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="ρ", thickness=14)
            ))
            corr_fig.update_layout(
                height=max(280, 42 * len(corr_symbols) + 60),
                margin=dict(l=10, r=10, t=20, b=10),
                template='plotly_white',
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(corr_fig, width="stretch")

        with corr_right:
            st.markdown(build_correlation_narrative(corr_symbols, corr_matrix), unsafe_allow_html=True)


# ----------------- Terminal header -----------------
st.markdown(
    f"""
    <div style='padding-bottom:8px; margin-bottom:10px; border-bottom:1px solid #e2e8f0;
                display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;'>
        <span style='font-size:17px; font-weight:700; color:#0f172a; letter-spacing:-0.3px;'>
            📈 AlphaRisk
        </span>
        <span style='font-size:12px; font-weight:500; color:#94a3b8;'>
            Quantitative Portfolio Terminal
        </span>
        <span style='font-size:11px; color:#64748b; margin-left:auto;'>
            S&amp;P 500 Constituents · {len(selected_symbols)} Active Assets · Benchmark SPY
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Idiosyncratic-risk warning banner
if abs_res is not None and abs_res['specific_var_ratio'] > 0.60:
    st.warning(f"⚠️ **Idiosyncratic Risk Warning**: non-systematic single-stock risk accounts for **{abs_res['specific_var_ratio']:.1%}** of total variance. The portfolio is too concentrated — most of its volatility comes from single-stock events that the six-factor model can't diversify away.")

st.write("")

# ----------------- Native left sidebar navigation -----------------
with st.sidebar:
    st.caption("NAVIGATION")
    for code, emoji, label in NAV_PAGES:
        is_current = (current_page == code)
        btn_label = f"{emoji}  {label}"
        if code == "holdings":
            btn_label += f"  ({len(selected_symbols)})"
        if st.button(
            btn_label, key=f"nav_{code}", width="stretch",
            type=("primary" if is_current else "secondary"),
            disabled=(not selected_symbols and code != "holdings")
        ):
            st.session_state[CURRENT_PAGE_KEY] = code
            st.rerun()

    st.divider()
    st.caption(f"{len(selected_symbols)} holdings ｜ weights auto-normalized to 100%")

# ----------------- Main content: renders the sub-page selected in the left nav -----------------
if current_page == "holdings":
    render_holdings_page()
elif current_page == "active":
    render_active_risk_page()
elif current_page == "absolute":
    render_absolute_risk_page()
elif current_page == "detail":
    render_detail_page()
