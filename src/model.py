# model.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_model(X, contamination=0.08, n_estimators=200, random_state=42):
    """Fits an Isolation Forest on the feature matrix X.
    Returns the fitted model — doesn't touch feature_df or the database.

    contamination=0.08 is what both main.py (live) and backtest.py actually
    pass in; this default just keeps them from silently diverging if a call
    site ever drops the explicit argument. It is NOT independently validated
    — if you want it empirically justified, sweep contamination against
    backtest.py's win-rate lift and log the result instead of hand-picking it.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    return model


def flag_anomalies(model, X):
    """Scores X with an already-fitted model.
    Returns (anomaly_labels, anomaly_scores) — -1 = anomaly, 1 = normal.
    Lower anomaly_score = more anomalous."""
    anomaly_labels = model.predict(X)
    anomaly_scores = model.decision_function(X)
    return anomaly_labels, anomaly_scores


def flag_anomalies_lof(X, contamination=0.08, n_neighbors=20):
    """Second, independent anomaly detector: Local Outlier Factor.

    Isolation Forest flags points that are easy to isolate with random
    axis-aligned splits. LOF flags points whose local neighborhood density is
    much lower than their neighbors' — a different mathematical notion of
    'unusual' (density-based, not partition-based). It does NOT replace the
    Isolation Forest; it's a cross-check. A trade flagged by both models is a
    much stronger signal than a trade flagged by only one, because it means
    the result isn't an artifact of one algorithm's particular quirks.

    LOF has no separate fit/predict step for scoring the same data it was fit
    on (that's the `novelty=False` mode, the default) — it fits and scores in
    one call via fit_predict, matching how the Isolation Forest is used in
    this pipeline (same-data fit_predict, no held-out set).

    Returns (anomaly_labels, anomaly_scores) — -1 = anomaly, 1 = normal.
    Lower anomaly_score = more anomalous, same convention as flag_anomalies().
    """
    n_neighbors = min(n_neighbors, max(2, len(X) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    anomaly_labels = lof.fit_predict(X)
    anomaly_scores = lof.negative_outlier_factor_
    return anomaly_labels, anomaly_scores


def explain_anomalies(X, feature_cols, top_n=3):
    """Cheap, model-agnostic explainability: for every row, finds the
    top_n features furthest from the population's typical value (median-based
    z-score, same robust approach as the size_zscore feature). Returns a list
    of human-readable strings, one per row, e.g.:
    'size_zscore (high, z=4.2); w_log_pvalue (high, z=3.8); w_age_days (low, z=-2.1)'

    This doesn't explain the Isolation Forest's internal splits — it explains
    *what made this trade unusual*, which is what a reviewer actually needs
    to triage a flag.
    """
    X = np.asarray(X, dtype=float)
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1e-9
    z = 0.6745 * (X - median) / mad

    explanations = []
    for row in z:
        top_idx = np.argsort(-np.abs(row))[:top_n]
        parts = [
            f"{feature_cols[i]} ({'high' if row[i] > 0 else 'low'}, z={row[i]:.1f})"
            for i in top_idx
        ]
        explanations.append("; ".join(parts))
    return explanations