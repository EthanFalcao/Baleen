# backtest.py
# Run standalone, manually or periodically: python backtest.py
#
# Independently fetches ALREADY-RESOLVED markets and their own trades,
# scores them with a model trained on that same closed-market data, and
# compares flagged vs. unflagged trades using two real statistical tests.
# Writes results into `backtest_stats`, which dashboard.py reads directly.
#
# Reports BOTH a raw win-rate comparison AND a calibration-adjusted "edge"
# comparison (actual outcome minus the market's own implied probability at
# trade time) -- not one instead of the other. Raw win rate is confounded by
# how skewed a given batch of markets' Yes/No outcomes happens to be (e.g. a
# batch where "No" wins 70% of the time will make anyone who bets "Yes" more
# look worse, regardless of skill); edge controls for that by comparing each
# trade to what its own price already implied. Reporting only whichever one
# looks better would be exactly the kind of cherry-picking this project's
# CLAUDE.md explicitly asks to avoid.

import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind

from db import get_connection
from data_fetch import get_closed_markets, get_trades
from features import FEATURE_COLS, engineer_features
from model import train_model, flag_anomalies, flag_anomalies_lof


def run_backtest():
    conn = get_connection()

    print("Fetching resolved markets for backtest...")
    closed_markets_df = get_closed_markets()

    print("Fetching trades for resolved markets...")
    all_trades = []
    for _, m in closed_markets_df.iterrows():
        trades = get_trades(m["condition_id"], m["question"])
        all_trades.extend(trades)
    trades_df = pd.DataFrame(all_trades).drop_duplicates(subset="trade_id")
    print(f"Fetched {len(trades_df)} trades from {len(closed_markets_df)} resolved markets")

    print("Engineering features...")
    feature_df = engineer_features(trades_df, closed_markets_df)

    print("Scoring (early trades only, >24h before resolution)...")
    early_mask = feature_df["hours_to_resolution"] > 24
    X_early = feature_df.loc[early_mask, FEATURE_COLS].values

    if len(X_early) < 10:
        print("Not enough early-window trades in resolved markets yet to score reliably.")
        conn.close()
        return

    model = train_model(X_early, contamination=0.08)
    anomaly_labels, anomaly_scores = flag_anomalies(model, X_early)
    lof_labels, lof_scores = flag_anomalies_lof(X_early, contamination=0.08)

    scored = feature_df.loc[early_mask].copy()
    scored["anomaly"] = anomaly_labels
    scored["anomaly_score"] = anomaly_scores
    scored["lof_anomaly"] = lof_labels
    scored["lof_anomaly_score"] = lof_scores
    scored["consensus_flag"] = ((scored["anomaly"] == -1) & (scored["lof_anomaly"] == -1)).astype(int)

    # Determine actual win/loss using the market's real final outcome
    scored = scored.merge(
        closed_markets_df[["condition_id", "final_yes_price"]], on="condition_id", how="left"
    )
    scored["is_yes_bet"] = scored["outcome"].str.lower().isin(["yes", "up"]).astype(int)
    scored["won"] = (
        ((scored["is_yes_bet"] == 1) & (scored["final_yes_price"] >= 0.99)) |
        ((scored["is_yes_bet"] == 0) & (scored["final_yes_price"] <= 0.01))
    ).astype(int)
    # "price" is the price of whichever token was actually traded, which IS
    # already the market's implied probability of that specific side winning
    # (no need to invert it for "No" bets -- each outcome's token is quoted
    # separately). edge = actual outcome minus what price implied you should
    # expect, so a trade that wins with a well-calibrated 20% price counts the
    # same as one that loses with an 80% price: both matched expectations.
    scored["edge"] = scored["won"] - scored["price"]

    flagged = scored[scored["anomaly"] == -1]
    unflagged = scored[scored["anomaly"] == 1]
    n1, n2 = len(flagged), len(unflagged)

    if n1 < 5 or n2 < 5:
        print(f"Not enough resolved, scored trades yet to test reliably "
              f"(flagged={n1}, unflagged={n2}). Need at least 5 in each group.")
        conn.close()
        return

    p1, p2 = flagged["won"].mean(), unflagged["won"].mean()

    p_pool = (flagged["won"].sum() + unflagged["won"].sum()) / (n1 + n2)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low, ci_high = (p1 - p2) - 1.96 * se_diff, (p1 - p2) + 1.96 * se_diff

    # Same comparison again, but on edge instead of raw win/loss -- this is
    # the calibration-adjusted version that controls for both market pricing
    # efficiency and whatever Yes/No skew this particular batch of markets
    # happens to have.
    e1, e2 = flagged["edge"].mean(), unflagged["edge"].mean()
    edge_t_stat, edge_p_value = ttest_ind(flagged["edge"], unflagged["edge"], equal_var=False)
    edge_se_diff = np.sqrt(flagged["edge"].std() ** 2 / n1 + unflagged["edge"].std() ** 2 / n2)
    edge_diff = e1 - e2
    edge_ci_low, edge_ci_high = edge_diff - 1.96 * edge_se_diff, edge_diff + 1.96 * edge_se_diff

    # Consensus check: does agreement between Isolation Forest and LOF
    # correlate with a stronger result than IF alone? If so, that's evidence
    # the primary result isn't an artifact of Isolation Forest's particular
    # splitting behavior. Reported alongside the main stat, never in place of
    # it — n_consensus is a subset of n_flagged, so it's a smaller, noisier
    # number and shouldn't be the headline figure.
    consensus = scored[scored["consensus_flag"] == 1]
    n_consensus = len(consensus)
    win_rate_consensus = consensus["won"].mean() if n_consensus >= 5 else None
    edge_consensus = consensus["edge"].mean() if n_consensus >= 5 else None

    stats_row = {
        "n_flagged": n1, "n_unflagged": n2,
        "win_rate_flagged": p1, "win_rate_unflagged": p2,
        "diff": p1 - p2, "ci_low": ci_low, "ci_high": ci_high,
        "z_stat": z_stat, "p_value": p_value,
        "edge_flagged": e1, "edge_unflagged": e2,
        "edge_diff": edge_diff, "edge_ci_low": edge_ci_low, "edge_ci_high": edge_ci_high,
        "edge_t_stat": edge_t_stat, "edge_p_value": edge_p_value,
        "n_consensus": n_consensus, "win_rate_consensus": win_rate_consensus,
        "edge_consensus": edge_consensus,
        "computed_at": pd.Timestamp.now("UTC").isoformat(),
    }
    stats = pd.DataFrame([stats_row])
    stats.to_sql("backtest_stats", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"\n--- Raw win rate (NOT calibration-adjusted -- confounded by this batch's Yes/No mix) ---")
    print(f"Flagged win rate:   {p1:.1%}  (n={n1})")
    print(f"Unflagged win rate: {p2:.1%}  (n={n2})")
    print(f"Difference: {p1 - p2:+.1%}   95% CI: [{ci_low:+.1%}, {ci_high:+.1%}]   p-value: {p_value:.4f}")

    print(f"\n--- Edge over implied probability (calibration-adjusted -- the trustworthy one) ---")
    print(f"Flagged mean edge:   {e1:+.4f}  (n={n1})")
    print(f"Unflagged mean edge: {e2:+.4f}  (n={n2})")
    print(f"Difference: {edge_diff:+.4f}   95% CI: [{edge_ci_low:+.4f}, {edge_ci_high:+.4f}]   p-value: {edge_p_value:.4f}")

    if win_rate_consensus is not None:
        print(f"\nConsensus (also flagged by LOF): win rate {win_rate_consensus:.1%}, "
              f"mean edge {edge_consensus:+.4f}  (n={n_consensus})")
    else:
        print(f"\nConsensus: not enough IF+LOF agreement yet to report (n={n_consensus}, need >= 5).")


if __name__ == "__main__":
    run_backtest()