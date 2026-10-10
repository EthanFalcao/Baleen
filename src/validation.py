# validation.py
# Run standalone: python validation.py
#
# Regression test for look-ahead bias in engineer_features(). Builds a small
# synthetic set of "early" trades, engineers features for them alone, then
# re-engineers features for the same early trades plus one deliberately
# extreme "future" trade appended after them. If any early trade's features
# differ between the two runs, engineer_features is leaking future
# information into past trades' scores — which is exactly the bug described
# in the project's P0 fix list.

import pandas as pd

from features import engineer_features, FEATURE_COLS

CONDITION_ID = "mkt_1"
BASE_TS = 1_700_000_000


def _synthetic_trades():
    rows = [
        {"trade_id": "t1", "condition_id": CONDITION_ID, "wallet": "wallet_a",
         "side": "BUY", "outcome": "Yes", "price": 0.50, "size": 100.0,
         "timestamp": BASE_TS, "market_title": "Test market"},
        {"trade_id": "t2", "condition_id": CONDITION_ID, "wallet": "wallet_b",
         "side": "BUY", "outcome": "Yes", "price": 0.52, "size": 120.0,
         "timestamp": BASE_TS + 3600, "market_title": "Test market"},
        {"trade_id": "t3", "condition_id": CONDITION_ID, "wallet": "wallet_a",
         "side": "SELL", "outcome": "Yes", "price": 0.55, "size": 90.0,
         "timestamp": BASE_TS + 7200, "market_title": "Test market"},
        {"trade_id": "t4", "condition_id": CONDITION_ID, "wallet": "wallet_c",
         "side": "BUY", "outcome": "No", "price": 0.40, "size": 60.0,
         "timestamp": BASE_TS + 10800, "market_title": "Test market"},
    ]
    # Deliberately extreme: huge size, near-certain price, from a wallet
    # (wallet_a) that already appears in the "early" set above — so this
    # single row can leak into BOTH market-level (size_percentile,
    # size_zscore) and wallet-level (w_* ) features if point-in-time isn't
    # enforced correctly.
    future_row = {
        "trade_id": "t_future", "condition_id": CONDITION_ID, "wallet": "wallet_a",
        "side": "BUY", "outcome": "Yes", "price": 0.99, "size": 500_000.0,
        "timestamp": BASE_TS + 100_000, "market_title": "Test market",
    }
    return rows, future_row


def _synthetic_market():
    return pd.DataFrame([{
        "condition_id": CONDITION_ID, "end_date": "2030-01-01T00:00:00Z",
        "liquidity": 10_000.0, "volume": 50_000.0, "best_bid": 0.50, "best_ask": 0.52,
    }])


def test_no_lookahead_bias():
    rows, future_row = _synthetic_trades()
    markets_df = _synthetic_market()

    feats_before = engineer_features(pd.DataFrame(rows), markets_df)
    feats_after = engineer_features(pd.DataFrame(rows + [future_row]), markets_df)

    early_ids = [r["trade_id"] for r in rows]
    before = feats_before.set_index("trade_id").loc[early_ids, FEATURE_COLS]
    after = feats_after.set_index("trade_id").loc[early_ids, FEATURE_COLS]

    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-9, atol=1e-9)
    print("PASS: engineer_features is point-in-time — adding a future trade "
          "did not change any earlier trade's features.")


if __name__ == "__main__":
    test_no_lookahead_bias()
