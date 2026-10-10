# features.py
import numpy as np
import pandas as pd
from scipy.stats import binomtest

FEATURE_COLS = [
    # Trade size features
    "size", "size_vs_liquidity", "size_vs_volume", "size_percentile",
    "size_zscore", "volume_spike_ratio",
    # Price features
    "price",
    # Time features
    "hours_to_resolution", "traded_in_final_48h", "traded_in_final_1h",
    # Trade context
    "is_buy", "is_yes_bet", "is_first_trade_in_market",
    # Wallet features
    "w_total_trades", "w_total_volume", "w_avg_size", "w_max_size",
    "w_markets_count", "w_buy_ratio", "w_age_days", "w_trades_per_market",
    "w_win_rate", "w_last_min_ratio", "w_avg_hours_to_res", "w_log_pvalue",
]

META_COLS = ["trade_id", "condition_id", "wallet", "side",
             "outcome", "price", "size", "timestamp", "market_title"]


def _compute_pvalue(row):
    n, k = int(row["w_resolved_trades"]), int(row["w_wins"])
    if n < 5:
        return 1.0
    return binomtest(k, n, p=0.5, alternative="greater").pvalue


def _pit_size_percentile_and_zscore(sizes):
    """Point-in-time percentile rank and modified z-score for each element of
    `sizes`, using ONLY the elements strictly before it in the array (i.e.
    trades that happened earlier in this market). The first trade in a market
    has no prior trades to compare against, so it gets NaN (filled to 0 later).
    """
    n = len(sizes)
    pct = np.full(n, np.nan)
    z = np.zeros(n)
    for i in range(1, n):
        prior = sizes[:i]
        pct[i] = (prior < sizes[i]).sum() / i
        if i >= 3:
            median = np.median(prior)
            mad = np.median(np.abs(prior - median))
            if mad > 0:
                z[i] = 0.6745 * (sizes[i] - median) / mad
    return pct, z


def _assign_pit_market_features(df):
    """Computes size_percentile/size_zscore per condition_id group and
    assigns them back onto df by index (pandas' groupby(...).apply(...)
    silently drops the grouping column in this pandas version when the
    function returns a full frame, so we build the columns manually)."""
    pct_all = pd.Series(index=df.index, dtype=float)
    z_all = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("condition_id"):
        pct, z = _pit_size_percentile_and_zscore(g["size"].to_numpy())
        pct_all.loc[g.index] = pct
        z_all.loc[g.index] = z
    df["size_percentile"] = pct_all
    df["size_zscore"] = z_all
    return df


def engineer_features(trades_df, markets_df):
    """Builds the feature matrix to feed into Isolation Forest.
    Pure function: takes trades + markets, returns feature_df.
    Doesn't touch the database or save any files."""

    df = trades_df.copy()
    df["trade_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    markets_df = markets_df.copy()
    markets_df["end_dt"] = pd.to_datetime(markets_df["end_date"], utc=True, errors="coerce")

    df = df.merge(markets_df[["condition_id", "end_dt", "liquidity", "volume"]],
                  on="condition_id", how="left")

    df["hours_to_resolution"] = ((df["end_dt"] - df["trade_dt"]).dt.total_seconds() / 3600).clip(lower=0)
    df["traded_in_final_48h"] = (df["hours_to_resolution"] < 48).astype(int)
    df["traded_in_final_1h"]  = (df["hours_to_resolution"] < 1).astype(int)

    df["size_vs_liquidity"] = df["size"] / df["liquidity"].clip(lower=1)
    df["size_vs_volume"]    = df["size"] / df["volume"].clip(lower=1)

    # NOTE: a "price_deviation" feature used to live here (|trade price -
    # mid_price| from best_bid/best_ask). Removed for the same reason as
    # price_move_favorable below: best_bid/best_ask are snapshotted at
    # PIPELINE RUNTIME, which for a closed market (backtest.py) is months
    # after resolution, so mid_price had already converged to the final
    # settlement price. Measured correlation with the actual win/loss
    # outcome: 0.86 -- almost as leaked as price_move_favorable's 0.94.

    df = df.sort_values(["condition_id", "timestamp"])
    rolling_mean = df.groupby("condition_id")["size"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    df["volume_spike_ratio"] = df["size"] / rolling_mean.clip(lower=1)
    # size_percentile / size_zscore: computed per-market, point-in-time — each
    # trade is only compared against trades that happened EARLIER in that same
    # market, never later ones (and never itself). See validation.py for the
    # regression test that guards this.
    df = _assign_pit_market_features(df)

    df["trade_num_in_market"]        = df.groupby(["wallet", "condition_id"]).cumcount() + 1
    df["is_first_trade_in_market"]   = (df["trade_num_in_market"] == 1).astype(int)

    df["is_yes_bet"]    = df["outcome"].str.lower().isin(["yes", "up"]).astype(int)
    df["is_buy"]        = (df["side"] == "BUY").astype(int)
    # NOTE: a "price_move_favorable" feature used to live here (mid_price vs.
    # the trade's own price). Removed -- mid_price comes from best_bid/best_ask
    # snapshotted at PIPELINE RUNTIME, which for a still-open market is "now"
    # (fine), but for a CLOSED market used in backtest.py is months after
    # resolution, so best_bid/best_ask have already converged to the final
    # settlement price. That made the feature correlate 0.94 with the actual
    # win/loss outcome in backtesting -- it was encoding the answer, not a
    # point-in-time signal. Confirmed via a supervised model that hit AUC 1.000
    # using this feature alone; see validation.py for the general point-in-time
    # regression test this class of bug should have been caught by, though it
    # specifically covers trade/wallet aggregates, not this feature's design.
    df["likely_winner"] = (((df["is_yes_bet"] == 1) & (df["price"] >= 0.85)) |
                            ((df["is_yes_bet"] == 0) & (df["price"] <= 0.15))).astype(int)
    df["near_resolved"] = ((df["price"] >= 0.85) | (df["price"] <= 0.15)).astype(int)

    # Wallet-level features: point-in-time — every aggregate below uses ONLY
    # this wallet's trades strictly before the trade being scored (shift(1)
    # before accumulating), never the trade itself and never later trades.
    # Sort is required for the shift-based expanding aggregates below; results
    # are assigned back by index so it doesn't matter that this reorders df's
    # rows relative to the condition_id/timestamp sort above.
    df = df.sort_values(["wallet", "timestamp"])
    grp = df.groupby("wallet")

    def _prior_cumsum(s):
        return s.shift(1).cumsum().fillna(0)

    w_trades_prior   = grp.cumcount()
    w_volume_prior   = grp["size"].transform(_prior_cumsum)
    w_max_prior      = grp["size"].transform(lambda s: s.shift(1).cummax().fillna(0))
    w_buycount_prior = grp["is_buy"].transform(_prior_cumsum)
    w_markets_prior  = grp["is_first_trade_in_market"].transform(_prior_cumsum)
    w_firstts_prior  = grp["timestamp"].transform(lambda s: s.shift(1).cummin())
    w_wins_prior     = grp["likely_winner"].transform(_prior_cumsum)
    w_resolved_prior = grp["near_resolved"].transform(_prior_cumsum)
    w_lastmin_prior  = grp["traded_in_final_1h"].transform(_prior_cumsum)
    w_hourssum_prior = grp["hours_to_resolution"].transform(_prior_cumsum)

    df["w_total_trades"]      = w_trades_prior
    df["w_total_volume"]      = w_volume_prior
    df["w_avg_size"]          = w_volume_prior / w_trades_prior.clip(lower=1)
    df["w_max_size"]          = w_max_prior
    df["w_markets_count"]     = w_markets_prior
    df["w_buy_ratio"]         = w_buycount_prior / w_trades_prior.clip(lower=1)
    df["w_age_days"]          = ((df["timestamp"] - w_firstts_prior.fillna(df["timestamp"])) / 86400).clip(lower=0)
    df["w_trades_per_market"] = w_trades_prior / w_markets_prior.clip(lower=1)
    df["w_wins"]               = w_wins_prior
    df["w_resolved_trades"]    = w_resolved_prior
    df["w_win_rate"]           = w_wins_prior / w_resolved_prior.clip(lower=1)
    df["w_last_min_ratio"]     = w_lastmin_prior / w_trades_prior.clip(lower=1)
    df["w_avg_hours_to_res"]   = w_hourssum_prior / w_trades_prior.clip(lower=1)
    df["w_binomial_pvalue"]    = df.apply(_compute_pvalue, axis=1)
    df["w_log_pvalue"]         = -np.log10(df["w_binomial_pvalue"].clip(lower=1e-10))

    df = df.sort_index()  # restore original trades_df row order

    all_cols = list(dict.fromkeys(META_COLS + FEATURE_COLS))
    feature_df = df[all_cols].copy()
    feature_df[FEATURE_COLS] = feature_df[FEATURE_COLS].fillna(0)

    return feature_df   # <-- just return it; main.py decides what to save