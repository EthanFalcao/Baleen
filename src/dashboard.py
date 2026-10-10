# dashboard.py
# Run with: streamlit run dashboard.py
# This file only reads from the database. It never fetches or scores anything.

import pandas as pd
import streamlit as st
import plotly.express as px

from features import FEATURE_COLS
from db import get_connection

st.set_page_config(
    page_title="Polymarket Whale Detection",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }
    h1 { font-size: 2.1rem; }
    h2, h3 { margin-top: 0.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Simple keyword-based market categorization ────────────────
# Sports is already filtered upstream in data_fetch.py; this just buckets
# what's left so you can toggle "does this cluster in geopolitics/macro?"
CATEGORY_KEYWORDS = {
    "Politics/Elections": ["election", "president", "senate", "congress", "governor", "vote", "primary"],
    "Macro/Fed":          ["fed ", "interest rate", "inflation", "cpi", "gdp", "recession", "rate hike", "rate cut", "fomc"],
    "Geopolitics":        ["iran", "israel", "russia", "ukraine", "china", "war", "ceasefire", "nato", "airspace", "invade", "strait of hormuz"],
    "Crypto":             ["bitcoin", "btc", "eth", "crypto", "ethereum", "solana", "coin"],
}


def categorize(title):
    t = str(title).lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return cat
    return "Other"


def _enrich(trades, markets):
    if len(trades):
        trades["trade_dt"] = pd.to_datetime(trades["timestamp"], unit="s", utc=True)
        trades["category"] = trades["market_title"].apply(categorize)
        if "slug" in markets.columns:
            trades = trades.merge(markets[["condition_id", "slug"]], on="condition_id", how="left")
        else:
            trades["slug"] = ""
        # Only real if the market was fetched after "slug" started being saved
        # (see data_fetch.py) -- older rows just won't have a working link.
        trades["market_url"] = trades["slug"].apply(
            lambda s: f"https://polymarket.com/event/{s}" if isinstance(s, str) and s else None
        )
    return trades


@st.cache_data(ttl=300)  # refresh cached data every 5 minutes
def load_global_counts():
    """Cheap aggregate counts for the header metrics -- no row data pulled.
    trade_features has grown into the hundreds of thousands of rows (that's
    the point: accumulated history), so `SELECT COUNT(*)` instead of loading
    every row just to call len() on it matters a lot here."""
    conn = get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM trade_features").fetchall()[0][0]
    total_markets = conn.execute("SELECT COUNT(DISTINCT condition_id) FROM trade_features").fetchall()[0][0]
    total_flagged = conn.execute("SELECT COUNT(*) FROM trade_features WHERE anomaly = -1").fetchall()[0][0]
    conn.close()
    return {
        "total_trades": total_trades,
        "total_markets": total_markets,
        "total_flagged": total_flagged,
        "flag_rate": (total_flagged / total_trades) if total_trades else 0,
    }


@st.cache_data(ttl=300)
def load_flagged_trades():
    """Only the flagged subset (~8% of the table). Every tab except Outlier
    Distribution only ever needs flagged trades, so this is what most of the
    dashboard is built on -- it's the row-level load that has to stay fast as
    trade_features keeps growing, unlike the old `SELECT *` which pulled the
    entire (unboundedly growing) table every time."""
    conn = get_connection()
    trades = pd.read_sql("SELECT * FROM trade_features WHERE anomaly = -1", conn)
    markets = pd.read_sql("SELECT * FROM markets", conn)
    conn.close()
    return _enrich(trades, markets), markets


@st.cache_data(ttl=300)
def load_distribution_sample(max_unflagged=20000):
    """For the Outlier Distribution tab only: ALL flagged trades plus a
    bounded, recent sample of unflagged ones, unflagged trades are ~92% of
    the table and growing, so pulling all of them isn't an option, but the
    histogram comparison doesn't need every point, just a representative
    (and, being recent, currently-relevant) sample of the "normal" shape."""
    cols = ", ".join(["anomaly", "anomaly_score"] + FEATURE_COLS)
    conn = get_connection()
    flagged = pd.read_sql(f"SELECT {cols} FROM trade_features WHERE anomaly = -1", conn)
    unflagged = pd.read_sql(
        f"SELECT {cols} FROM trade_features WHERE anomaly = 1 "
        f"ORDER BY timestamp DESC LIMIT {int(max_unflagged)}",
        conn,
    )
    conn.close()
    combined = pd.concat([flagged, unflagged], ignore_index=True)
    combined["Flag"] = combined["anomaly"].map({-1: "Flagged", 1: "Normal"})
    return combined


@st.cache_data(ttl=300)
def load_backtest_stats():
    conn = get_connection()
    try:
        backtest_stats = pd.read_sql("SELECT * FROM backtest_stats", conn)
    except Exception:
        backtest_stats = pd.DataFrame()
    conn.close()
    return backtest_stats


def _prep_display(df, ref_len):
    """Adds a short wallet label and a 0-100 severity score (100 = most anomalous)
    so tables are scannable without reading raw anomaly_score decimals."""
    df = df.copy()
    df["wallet_short"] = df["wallet"].str[:6] + "…" + df["wallet"].str[-4:]
    if ref_len:
        rank_pct = df["anomaly_score"].rank(pct=True, ascending=True)
        df["severity"] = ((1 - rank_pct) * 100).round(0)
    else:
        df["severity"] = 0
    return df


global_counts = load_global_counts()
flagged_trades, markets = load_flagged_trades()
backtest_stats = load_backtest_stats()

# ── Header ────────────────────────────────────────────────────
st.title("Polymarket Whale Detection")
st.caption(
    "Flags Polymarket trades that look unusual for their size, timing, and the "
    "wallet's history. A flag is a lead to look into, not proof of insider "
    "trading. Someone who reads the news well can look the same as someone "
    "with inside information, so every flag still needs a person to check it."
)

# ── Sidebar filters ───────────────────────────────────────────
# Everything below filters the FLAGGED-trade views only (wallets, all-flagged,
# history) -- the header metrics above are all-time totals across the whole
# table and don't move with these filters. See load_flagged_trades()'s
# docstring for why: trade_features has grown into the hundreds of thousands
# of rows, and only the flagged ~8% of it is small enough to load per-request.
latest_run = flagged_trades["run_timestamp"].max() if len(flagged_trades) else None

with st.sidebar:
    st.header("Filters")
    st.caption(f"Last pipeline run:\n\n**{latest_run}**" if latest_run else "No runs yet.")
    st.caption("These filters change the Wallets, All Flagged Trades, and History "
               "tabs. The totals at the top always show everything.")

    if len(flagged_trades):
        min_d, max_d = flagged_trades["trade_dt"].min().date(), flagged_trades["trade_dt"].max().date()
        if min_d == max_d:
            date_range = (min_d, max_d)
            st.caption(f"All flagged trades are from {min_d}.")
        else:
            date_range = st.date_input(
                "Trade date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
            )
    else:
        date_range = None

    categories = sorted(flagged_trades["category"].unique()) if len(flagged_trades) else []
    selected_categories = st.multiselect(
        "Market category", categories, default=categories,
        help="Check whether flags bunch up in one type of market, like geopolitics.",
    )

    min_severity = st.slider(
        "Minimum severity",
        min_value=0, max_value=100, value=0,
        help="Severity runs from 0 to 100, where 100 is the most unusual trade. "
             "Leave it at 0 to see every flagged trade, or raise it to keep only "
             "the strongest flags.",
    )

# ── Apply filters once, use everywhere ────────────────────────
flagged_f = flagged_trades.copy()
if len(flagged_f) and date_range and len(date_range) == 2:
    start, end = date_range
    flagged_f = flagged_f[(flagged_f["trade_dt"].dt.date >= start) & (flagged_f["trade_dt"].dt.date <= end)]
if len(flagged_f) and selected_categories:
    flagged_f = flagged_f[flagged_f["category"].isin(selected_categories)]

flagged_all = _prep_display(flagged_f, len(flagged_f))
current_flagged = flagged_all[flagged_all["run_timestamp"] == latest_run].copy()

# ── Top-line metrics (all-time totals -- see sidebar note above) ──────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades tracked", f"{global_counts['total_trades']:,}")
c2.metric("Markets monitored", f"{global_counts['total_markets']:,}")
c3.metric("Flagged (all-time)", f"{global_counts['total_flagged']:,}")
c4.metric("Flag rate", f"{global_counts['flag_rate']:.1%}")
if len(flagged_all) < len(flagged_trades):
    st.caption(f"Filters on: showing {len(flagged_all):,} of "
               f"{len(flagged_trades):,} flagged trades.")

st.divider()

# ── Methodology & validated finding ──────────────────────────
left, right = st.columns([1.3, 1])

with left:
    st.subheader("How this works")
    st.markdown(
        "- Each trade gets scored by an **Isolation Forest** using its size "
        "compared to the market, how long before resolution it was placed, and "
        "the wallet's past activity (win rate, how often it trades, buy/sell mix).\n"
        "- A second model, **Local Outlier Factor**, scores the same trades as a "
        "cross-check.\n"
        "- Only trades placed **more than 48 hours before the market resolves** "
        "get scored. Late trades mostly reflect news everyone already has.\n"
        "- A flag means the trade is *unusual*. It doesn't mean anything illegal "
        "happened."
    )

with right:
    st.subheader("Backtest result")
    if len(backtest_stats) and "edge_diff" in backtest_stats.columns:
        s = backtest_stats.iloc[0]
        sig = "statistically significant" if s["edge_p_value"] < 0.05 else "not statistically significant"
        direction = "beat" if s["edge_diff"] > 0 else "trailed"
        st.info(
            f"On resolved markets, flagged trades **{direction} unflagged trades "
            f"by {abs(s['edge_diff']):.1%}** on edge.\n\n"
            f"Edge is the trade's result minus the price paid. The price is what "
            f"the market thought the odds were, so edge shows whether a trade did "
            f"better than the market expected.\n\n"
            f"Flagged: {s['edge_flagged']:+.3f} average edge (n={int(s['n_flagged'])}). "
            f"Unflagged: {s['edge_unflagged']:+.3f} (n={int(s['n_unflagged'])}).\n\n"
            f"95% CI [{s['edge_ci_low']:+.3f}, {s['edge_ci_high']:+.3f}], "
            f"p = {s['edge_p_value']:.4f} ({sig})."
        )
        with st.expander("Why not just compare win rates?"):
            raw_sig = "statistically significant" if s["p_value"] < 0.05 else "not statistically significant"
            st.markdown(
                f"Flagged trades won **{s['win_rate_flagged']:.1%}** of the time and "
                f"unflagged trades won **{s['win_rate_unflagged']:.1%}** (difference "
                f"{s['diff']:+.1%}, p = {s['p_value']:.4f}, {raw_sig}). That number "
                f"is misleading on its own. If most markets in a batch resolve No, "
                f"anyone who buys Yes more often will look worse no matter how good "
                f"they are. Edge adjusts for that because it compares each trade to "
                f"its own odds. Both numbers are shown so you can see the difference."
            )
        st.caption(f"Last backtest: {s['computed_at']}. The backtest reruns every day "
                   f"on a new set of resolved markets, so this number moves around.")
        if "win_rate_consensus" in s and pd.notna(s["win_rate_consensus"]):
            st.caption(
                f"Trades flagged by both models (Isolation Forest and Local Outlier "
                f"Factor) averaged **{s['edge_consensus']:+.3f}** edge "
                f"(n={int(s['n_consensus'])})."
            )
    elif len(backtest_stats):
        s = backtest_stats.iloc[0]
        sig = "statistically significant" if s["p_value"] < 0.05 else "not statistically significant"
        st.info(
            f"Flagged trades won **{s['win_rate_flagged']:.1%}** of the time "
            f"(n={int(s['n_flagged'])}) and unflagged trades won "
            f"**{s['win_rate_unflagged']:.1%}** (n={int(s['n_unflagged'])}).\n\n"
            f"Difference: {s['diff']:+.1%}, 95% CI [{s['ci_low']:+.1%}, {s['ci_high']:+.1%}], "
            f"p = {s['p_value']:.4f} ({sig})."
        )
        st.caption(f"Last backtest: {s['computed_at']} (older format, before the edge metric was added).")
    else:
        st.warning(
            "No backtest results yet. Run `python backtest.py` once some flagged "
            "markets have resolved."
        )

st.divider()

# ── Tabs: wallet-first narrative ──────────────────────────────
tab_wallets, tab_dist, tab_current, tab_history = st.tabs(
    ["Suspicious Wallets", "Score Distribution", "All Flagged Trades", "History"]
)

# --- Tab 1: Wallet-level rollup ----------
with tab_wallets:
    st.subheader("Wallets with the most flags")
    st.caption(
        "Each row adds up one wallet's flagged trades within your filters. A "
        "wallet that keeps getting flagged across different markets is worth "
        "more attention than a single odd trade."
    )

    if len(flagged_all):
        wallet_view = (
            flagged_all.groupby("wallet")
            .agg(
                flagged_trades=("trade_id", "count"),
                consensus_trades=("consensus_flag", "sum"),
                total_flagged_volume=("size", "sum"),
                avg_anomaly_score=("anomaly_score", "mean"),
                worst_anomaly_score=("anomaly_score", "min"),
                markets_touched=("condition_id", "nunique"),
            )
            .reset_index()
            .sort_values("worst_anomaly_score")
        )
        wallet_view["total_flagged_volume"] = wallet_view["total_flagged_volume"].round(0)
        wallet_view["avg_anomaly_score"] = wallet_view["avg_anomaly_score"].round(4)
        wallet_view["worst_anomaly_score"] = wallet_view["worst_anomaly_score"].round(4)
        wallet_view["wallet_short"] = wallet_view["wallet"].str[:6] + "…" + wallet_view["wallet"].str[-4:]

        st.dataframe(
            wallet_view[["wallet_short", "flagged_trades", "consensus_trades", "total_flagged_volume",
                         "avg_anomaly_score", "worst_anomaly_score", "markets_touched"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "wallet_short": st.column_config.TextColumn("Wallet"),
                "flagged_trades": st.column_config.NumberColumn("Flagged trades"),
                "consensus_trades": st.column_config.NumberColumn(
                    "Flagged by both models",
                    help="How many of this wallet's flags the second model (Local Outlier Factor) also flagged.",
                ),
                "total_flagged_volume": st.column_config.NumberColumn("Flagged volume ($)", format="$%,.0f"),
                "avg_anomaly_score": st.column_config.NumberColumn("Avg anomaly score"),
                "worst_anomaly_score": st.column_config.NumberColumn("Worst anomaly score"),
                "markets_touched": st.column_config.NumberColumn("Markets touched"),
            },
        )

        st.markdown("#### Look at one wallet")
        chosen_wallet = st.selectbox("Wallet", wallet_view["wallet"])
        wallet_trades = flagged_all[flagged_all["wallet"] == chosen_wallet].sort_values("timestamp")
        st.dataframe(
            wallet_trades[["market_title", "market_url", "side", "outcome", "size", "price",
                           "severity", "why_flagged", "run_timestamp"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "market_url": st.column_config.LinkColumn("Polymarket", display_text="Open ↗"),
                "size": st.column_config.NumberColumn("Size ($)", format="$%,.0f"),
                "price": st.column_config.NumberColumn("Price", format="%.2f"),
                "severity": st.column_config.ProgressColumn(
                    "Severity", format="%.0f", min_value=0, max_value=100
                ),
                "why_flagged": st.column_config.TextColumn("Why flagged", width="large"),
            },
        )
    else:
        st.write("No flagged trades match these filters.")

# --- Tab 2: Score distribution --------------------------------
with tab_dist:
    st.subheader("Anomaly scores: flagged vs. normal")
    st.caption(
        "Lower scores are more unusual, so flagged trades should sit in their "
        "own tail on the left. The normal group is the 20,000 most recent "
        "unflagged trades, not all of them, since loading every trade would be "
        "too slow. The sidebar filters don't apply here."
    )

    scored = load_distribution_sample()
    if len(scored):
        fig = px.histogram(
            scored, x="anomaly_score", color="Flag", nbins=60,
            color_discrete_map={"Flagged": "#e74c3c", "Normal": "#3498db"},
            opacity=0.8, barmode="overlay",
        )
        fig.update_layout(
            margin=dict(t=10, b=10), height=340,
            xaxis_title="Anomaly score (lower = more anomalous)",
            yaxis_title="Trade count",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature breakdown")
        st.caption(
            "Pick a feature to compare flagged and normal trades on it. This is "
            "the same information behind each trade's \"Why flagged\" note."
        )
        numeric_feats = [c for c in FEATURE_COLS if c in scored.columns and scored[c].notna().any()]
        if numeric_feats:
            default_idx = numeric_feats.index("size_zscore") if "size_zscore" in numeric_feats else 0
            feat = st.selectbox("Feature", numeric_feats, index=default_idx)
            fig2 = px.histogram(
                scored, x=feat, color="Flag", nbins=60,
                color_discrete_map={"Flagged": "#e74c3c", "Normal": "#3498db"},
                opacity=0.8, barmode="overlay",
            )
            fig2.update_layout(margin=dict(t=10, b=10), height=340, legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(
                "No feature values are stored for these trades yet. They'll show "
                "up after the next pipeline run."
            )
    else:
        st.write("No scored trades yet.")

# --- Tab 3: all flagged trades (slider is optional, not a gate) ---
with tab_current:
    st.subheader("All flagged trades")
    st.caption(
        "Every flagged trade in your filters, most unusual first. Use the "
        "severity slider in the sidebar if the list is too long."
    )

    only_latest_run = st.checkbox(
        "Only trades first seen in the latest run",
        value=False,
        help="A trade is saved the first time the pipeline sees it and never "
             "rewritten, so this only shows trades that were new in the latest "
             "run. The list can look short because of that.",
    )
    base_view = current_flagged if only_latest_run else flagged_all
    view = base_view[base_view["severity"] >= min_severity].sort_values("anomaly_score")

    if len(base_view):
        st.caption(f"Highest severity in this list: **{base_view['severity'].max():.0f}** / 100.")

    if len(view):
        st.dataframe(
            view[["wallet_short", "market_title", "market_url", "size", "side", "outcome",
                 "price", "severity", "why_flagged"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "wallet_short": st.column_config.TextColumn("Wallet"),
                "market_url": st.column_config.LinkColumn("Polymarket", display_text="Open ↗"),
                "size": st.column_config.NumberColumn("Size ($)", format="$%,.0f"),
                "price": st.column_config.NumberColumn("Price", format="%.2f"),
                "severity": st.column_config.ProgressColumn(
                    "Severity", format="%.0f", min_value=0, max_value=100
                ),
                "why_flagged": st.column_config.TextColumn("Why flagged", width="large"),
            },
        )
    else:
        st.write(f"No flagged trades at severity {min_severity} or above. Try lowering the slider.")

# --- Tab 4: history -----------------------------------
with tab_history:
    st.subheader("Flagged trades per run")

    if len(flagged_all):
        flagged_all["run_dt"] = pd.to_datetime(flagged_all["run_timestamp"], utc=True)

        trend = (
            flagged_all
            .groupby(pd.Grouper(key="run_dt", freq="6h"))
            .size()
            .reset_index(name="flagged_count")
        )
        trend = trend[trend["flagged_count"] > 0]  # drop empty buckets

        fig = px.bar(trend, x="run_dt", y="flagged_count")
        fig.update_layout(
            margin=dict(t=10, b=10), height=350,
            xaxis_title="", yaxis_title="Flagged trades",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full log")
        st.dataframe(
            flagged_all.sort_values("run_timestamp", ascending=False)[
                ["run_timestamp", "wallet_short", "market_title", "market_url", "size", "side",
                 "outcome", "price", "severity"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "wallet_short": st.column_config.TextColumn("Wallet"),
                "market_url": st.column_config.LinkColumn("Polymarket", display_text="Open ↗"),
                "size": st.column_config.NumberColumn("Size ($)", format="$%,.0f"),
                "price": st.column_config.NumberColumn("Price", format="%.2f"),
                "severity": st.column_config.ProgressColumn(
                    "Severity", format="%.0f", min_value=0, max_value=100
                ),
            },
        )
    else:
        st.write("No flagged trades yet.")

st.divider()
st.caption(
    "Data from the Polymarket Gamma and Data APIs. Models: Isolation Forest, "
    "cross-checked with Local Outlier Factor. Only trades placed more than 48 "
    "hours before resolution are scored."
)
