# main.py
import pandas as pd
from datetime import datetime, timezone

from features import FEATURE_COLS, engineer_features
from db import get_connection, save_markets, save_flagged_trades
from data_fetch import get_active_markets, get_trades
from model import train_model, flag_anomalies, flag_anomalies_lof, explain_anomalies


def run_pipeline():
    conn = get_connection()

    print("Fetching active markets...")
    markets_df = get_active_markets()
    save_markets(markets_df, conn)

    print("Fetching trades...")
    all_trades = []
    for _, m in markets_df.iterrows():
        trades = get_trades(m["condition_id"], m["question"])
        all_trades.extend(trades)
        print(f"  {m['question'][:60]}... -> {len(trades)} trades")
    trades_df = pd.DataFrame(all_trades).drop_duplicates(subset="trade_id")
    print(f"Fetched {len(trades_df)} total trades")

    print("Engineering features...")
    feature_df = engineer_features(trades_df, markets_df)

    print("Scoring with Isolation Forest (early trades only, >48h before resolution)...")

    early_mask = feature_df["hours_to_resolution"] > 48
    X_early = feature_df.loc[early_mask, FEATURE_COLS].values

    model = train_model(X_early, contamination=0.08)

    feature_df["anomaly"] = 0
    feature_df["anomaly_score"] = 0.0
    feature_df["why_flagged"] = ""
    feature_df["lof_anomaly"] = 0
    feature_df["lof_anomaly_score"] = 0.0
    feature_df["consensus_flag"] = 0

    anomaly_labels, anomaly_scores = flag_anomalies(model, X_early)
    feature_df.loc[early_mask, "anomaly"] = anomaly_labels
    feature_df.loc[early_mask, "anomaly_score"] = anomaly_scores
    feature_df.loc[early_mask, "why_flagged"] = explain_anomalies(X_early, FEATURE_COLS)

    # Cross-check with a second, differently-shaped anomaly detector (see
    # model.py's flag_anomalies_lof docstring) — doesn't change what gets
    # flagged, just marks which flags both models agree on.
    lof_labels, lof_scores = flag_anomalies_lof(X_early, contamination=0.08)
    feature_df.loc[early_mask, "lof_anomaly"] = lof_labels
    feature_df.loc[early_mask, "lof_anomaly_score"] = lof_scores
    feature_df.loc[early_mask, "consensus_flag"] = (
        (feature_df.loc[early_mask, "anomaly"] == -1) & (lof_labels == -1)
    ).astype(int)

    feature_df["run_timestamp"] = datetime.now(timezone.utc).isoformat()

    save_flagged_trades(feature_df, conn)

    flagged = feature_df[feature_df["anomaly"] == -1].sort_values("anomaly_score")
    consensus = feature_df[feature_df["consensus_flag"] == 1]
    print(f"Done. Flagged {len(flagged)} / {len(feature_df)} trades "
          f"({len(flagged)/len(feature_df):.1%}); "
          f"{len(consensus)} also flagged by the LOF cross-check.")

    conn.close()


if __name__ == "__main__":
    run_pipeline()