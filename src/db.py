# db.py
import libsql
from config import TURSO_DATABASE_URL, TURSO_AUTH_TOKEN
from features import FEATURE_COLS

_BASE_COLS = {
    "trade_id", "condition_id", "wallet", "side", "outcome", "price", "size",
    "timestamp", "market_title", "anomaly", "anomaly_score", "why_flagged", "run_timestamp",
    "lof_anomaly", "lof_anomaly_score", "consensus_flag",
}
# FEATURE_COLS includes "price" and "size", which are already declared above —
# only add the feature columns that aren't already part of the base schema.
_EXTRA_FEATURE_COLS = [c for c in FEATURE_COLS if c not in _BASE_COLS]
_FEATURE_COLS_SQL = ",\n            ".join(f"{c} REAL" for c in _EXTRA_FEATURE_COLS)


def get_connection():
    """Opens a connection to the Turso (hosted, SQLite-compatible) database.
    Creates tables only if they don't already exist —
    never drops them, so history persists across runs."""
    conn = libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT, slug TEXT, end_date TEXT, volume REAL, liquidity REAL,
            best_bid REAL, best_ask REAL, yes_price REAL, no_price REAL,
            last_trade_price REAL, yes_token_id TEXT, no_token_id TEXT
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS trade_features (
            trade_id TEXT PRIMARY KEY, condition_id TEXT, wallet TEXT,
            side TEXT, outcome TEXT, price REAL, size REAL,
            timestamp INTEGER, market_title TEXT,
            anomaly INTEGER, anomaly_score REAL, why_flagged TEXT, run_timestamp TEXT,
            lof_anomaly INTEGER, lof_anomaly_score REAL, consensus_flag INTEGER,
            {_FEATURE_COLS_SQL}
        )
    """)
    conn.commit()

    # Migration: add ANY core column an older committed database might be missing
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(trade_features)").fetchall()]
    core_cols = {
        "anomaly": "INTEGER", "anomaly_score": "REAL",
        "why_flagged": "TEXT", "run_timestamp": "TEXT",
        "lof_anomaly": "INTEGER", "lof_anomaly_score": "REAL", "consensus_flag": "INTEGER",
    }
    for col, sqltype in core_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE trade_features ADD COLUMN {col} {sqltype}")
            conn.commit()
    for col in FEATURE_COLS:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE trade_features ADD COLUMN {col} REAL")
            conn.commit()

    # trade_features has grown into the hundreds of thousands of rows (it's
    # designed to keep growing -- that's the point, accumulated history) and
    # dashboard.py filters/sorts it by these two columns on every load. Without
    # indexes each of those is a full table scan; this keeps it scaling as the
    # table keeps growing instead of getting slower every month.
    # Composite (not two single-column indexes): dashboard.py's queries filter
    # by anomaly AND sort by timestamp together, and a composite index lets
    # SQLite satisfy both from one sorted range scan. A separate single-column
    # timestamp index actually made that query SLOWER (measured 4.2s -> 11.8s)
    # -- it picked an index-then-row-lookup plan that's worse than a scan when
    # each lookup is a network round trip to Turso.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_features_anomaly_ts ON trade_features(anomaly, timestamp)")
    conn.commit()

    print("DB ready.")
    return conn


def save_markets(markets_df, conn):
    """Overwrites the markets table each run — this is a CURRENT
    snapshot of active markets, not something we need history for."""
    markets_df.to_sql("markets", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"Saved {len(markets_df)} markets.")


def save_flagged_trades(feature_df, conn):
    """Appends new trade/anomaly results  this IS the history we want
    to keep. INSERT OR IGNORE skips any trade_id already saved from a
    previous run, so re-running the pipeline never creates duplicates."""
    
    # Get actual columns from the database
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(trade_features)").fetchall()]
    
    # Build column list: only include columns that exist in both DataFrame and database
    cols = list(dict.fromkeys(
        ["trade_id", "condition_id", "wallet", "side", "outcome",
         "price", "size", "timestamp", "market_title",
         "anomaly", "anomaly_score", "why_flagged", "run_timestamp",
         "lof_anomaly", "lof_anomaly_score", "consensus_flag"] + FEATURE_COLS
    ))
    
    # Filter to only columns that exist in the database
    cols = [c for c in cols if c in existing_cols]
    
    # Create staging table with only the columns we'll actually insert
    feature_df[cols].to_sql("trade_features_staging", conn, if_exists="replace", index=False)
    
    # Insert with explicit column list
    cols_str = ", ".join(cols)
    conn.execute(f"""
        INSERT OR IGNORE INTO trade_features ({cols_str})
        SELECT {cols_str} FROM trade_features_staging
    """)
    conn.execute("DROP TABLE trade_features_staging")
    conn.commit()
    print(f"Saved {len(feature_df)} trade records (duplicates skipped automatically).")



'''
the old code dropped (deleted) all four tables every single time it ran, 
then recreated them empty. That's fine for a notebook you're actively editing
and re-running by hand, but it's a real bug for something meant to run automatically 
on a schedule
'''
