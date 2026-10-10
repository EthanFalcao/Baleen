# data_fetch.py
import requests
import json
import time
import pandas as pd

from config import GAMMA_API, DATA_API, HEADERS, TOP_N_MARKETS, TRADES_PER_MARKET

SPORTS_KEYWORDS = [
    "nba", "nfl", "nhl", "mlb", "soccer", "football", "basketball",
    "baseball", "hockey", "tennis", "golf", "ufc", "mma", "wrestling",
    "fifa", "wimbledon", "superbowl", "champions league", "premier league",
    "score", "match", "game", "playoff", "championship", "tournament",
    "over/under", "total kills", "lol-dnf", " vs ", " vs. ", 'nomination', 'election', 'lebron'
]


def is_sports(market):
    text = (market.get("question", "") + market.get("slug", "")).lower()
    return any(kw in text for kw in SPORTS_KEYWORDS)


def get_active_markets():
    """Fetches currently active, non-sports markets. Returns a DataFrame.
    Does NOT touch the database — main.py decides what to save."""
    r = requests.get(
        f"{GAMMA_API}/markets",
        params={"active": "true", "closed": "false",
                "limit": TOP_N_MARKETS, "order": "volume1wk", "ascending": "false"},
        headers=HEADERS,
        timeout=15,
    )
    raw_markets = r.json()
    raw_markets = [m for m in raw_markets if not is_sports(m)]

    markets = []
    for m in raw_markets:
        token_ids      = json.loads(m.get("clobTokenIds", "[]"))
        outcome_prices = json.loads(m.get("outcomePrices", '["0.5","0.5"]'))

        markets.append({
            "condition_id":     m.get("conditionId", ""),
            "question":         m.get("question", ""),
            "slug":             m.get("slug", ""),
            "end_date":         m.get("endDate", ""),
            "volume":           float(m.get("volume") or 0),
            "liquidity":        float(m.get("liquidityNum") or 0),
            "yes_price":        float(outcome_prices[0]),
            "no_price":         float(outcome_prices[1]),
            "best_bid":         float(m.get("bestBid") or 0),
            "best_ask":         float(m.get("bestAsk") or 0),
            "last_trade_price": float(m.get("lastTradePrice") or 0),
            "yes_token_id":     token_ids[0] if token_ids else "",
            "no_token_id":      token_ids[1] if len(token_ids) > 1 else "",
        })

    markets_df = pd.DataFrame(markets)
    print(f"Fetched {len(markets_df)} markets.")
    return markets_df


def get_trades(condition_id, question=""):
    """Fetches all trades for ONE market (given its condition_id).
    Returns a list of trade dicts. Does NOT touch the database."""
    trades = []
    offset = 0

    while offset < TRADES_PER_MARKET:
        r = requests.get(
            f"{DATA_API}/trades",
            params={"market": condition_id, "limit": 500, "offset": offset},
            headers=HEADERS,
            timeout=15,
        )
        if not r.ok:
            break

        page = r.json()
        if not page:
            break

        for t in page:
            tx    = t.get("transactionHash", "")
            asset = t.get("asset", "")
            trades.append({
                "trade_id":     f"{tx[:20]}_{asset[:10]}",
                "condition_id": condition_id,
                "wallet":       t.get("proxyWallet", ""),
                "side":         t.get("side", ""),
                "outcome":      t.get("outcome", ""),
                "price":        float(t.get("price") or 0),
                "size":         float(t.get("size") or 0),
                "timestamp":    int(t.get("timestamp") or 0),
                "tx_hash":      tx,
                "market_title": t.get("title", "") or question,
            })

        if len(page) < 500:
            break
        offset += 500
        time.sleep(0.2)

    time.sleep(0.4)
    return trades


def get_closed_markets(max_markets=None):
    """Fetches already-resolved markets, for backtesting. Returns a DataFrame.
    Paginates through multiple pages since the API caps results per request."""
    max_markets = max_markets or TOP_N_MARKETS
    all_raw = []
    offset = 0
    page_size = 100

    while len(all_raw) < max_markets:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"closed": "true", "limit": page_size, "offset": offset,
                    "order": "volume", "ascending": "false"},
            headers=HEADERS,
            timeout=15,
        )
        page = r.json()
        if not page:
            break
        all_raw.extend(page)
        offset += page_size
        if len(page) < page_size:
            break

    raw_closed_markets = [m for m in all_raw if not is_sports(m)]

    closed_markets = []
    for m in raw_closed_markets:
        token_ids      = json.loads(m.get("clobTokenIds", "[]"))
        outcome_prices = json.loads(m.get("outcomePrices", '["0.5","0.5"]'))

        closed_markets.append({
            "condition_id":     m.get("conditionId", ""),
            "question":         m.get("question", ""),
            "slug":             m.get("slug", ""),
            "end_date":         m.get("endDate", ""),
            "volume":           float(m.get("volume") or 0),
            "liquidity":        float(m.get("liquidityNum") or 0),
            "best_bid":         float(m.get("bestBid") or 0),
            "best_ask":         float(m.get("bestAsk") or 0),
            "final_yes_price":  float(outcome_prices[0]),
            "final_no_price":   float(outcome_prices[1]),
            "yes_token_id":     token_ids[0] if token_ids else "",
            "no_token_id":      token_ids[1] if len(token_ids) > 1 else "",
        })

    closed_markets_df = pd.DataFrame(closed_markets)
    print(f"Fetched {len(closed_markets_df)} closed markets.")
    return closed_markets_df
