# Polymarket Whale Detection

This project looks for unusual trades on Polymarket that could come from people trading on information the rest of the market doesn't have yet. It pulls trades every day, scores them with anomaly detection, and shows the results on a Streamlit dashboard: [baleen.streamlit.app](https://baleen.streamlit.app)

A flag is a lead, not proof. The model can't tell an insider apart from someone who just reads the news well.

## How it works

1. **Pull data.** Market info comes from Polymarket's Gamma API and trades come from the Data API. Sports markets are skipped.
2. **Build features.** For each trade: its size compared to the market's liquidity and volume, how long before resolution it was placed, and the wallet's history (win rate, number of trades, how long it's been active). Every feature only uses trades that happened before the one being scored, so nothing from the future leaks in.
3. **Score trades.** An Isolation Forest flags the most unusual 8% of trades placed more than 48 hours before resolution. A Local Outlier Factor model scores the same trades as a second opinion.
4. **Check against real outcomes.** `backtest.py` runs the same steps on markets that have already resolved and compares flagged trades to unflagged ones.
5. **Show it.** The dashboard reads from the database and lists flagged trades and wallets, with links back to each market on Polymarket.

A GitHub Actions job runs the pipeline and the backtest once a day. Data is stored in Turso (hosted SQLite).

## Results so far

No reliable signal yet.

An early notebook version showed flagged trades winning 71% of the time vs. 42% for unflagged ones. That number was wrong. Some features were calculated using trades that happened after the trade being scored, and two others used order book prices captured after the markets had resolved. With those bugs fixed, the result flipped: flagged trades won less often than unflagged ones.

Raw win rate also turned out to be the wrong thing to measure. In most samples the majority of markets resolve No, so anyone who buys Yes more often looks worse no matter how good they are. The backtest now also reports **edge**: the trade's result minus the price paid. Since the price is the market's odds at the time, edge shows whether a trade did better than the market expected.

On edge, flagged trades came out 3.6 percentage points ahead on one sample of resolved markets and 5.0 points behind on a later one. Each result was statistically significant on its own, which mostly tells me the samples are too small and too different from each other to trust yet.

## Project layout

| File | What it does |
|---|---|
| `src/data_fetch.py` | Pulls markets and trades from the Polymarket APIs |
| `src/features.py` | Builds the per-trade features |
| `src/model.py` | Isolation Forest, Local Outlier Factor, and the "why flagged" notes |
| `src/main.py` | Daily run on active markets |
| `src/backtest.py` | Daily run on resolved markets, compares flagged vs. unflagged |
| `src/validation.py` | Test that checks features don't use future data |
| `src/db.py` | Turso connection and tables |
| `src/dashboard.py` | Streamlit dashboard |
| `.github/workflows/polymarket-pipeline.yml` | Runs `main.py` and `backtest.py` every day |
| `polymarket_notebook.ipynb` | Early exploration. Its numbers are out of date. |

## Known issues and next steps

- **Too many flags.** The model flags 8% of trades, which comes out to thousands a day. A ranked top 10 or top 20 would be more useful.
- **The backtest moves around.** It runs on a different set of resolved markets each day and overwrites the previous result. It should use a larger, fixed set.
- **No confirmed insider cases to test against.** ForesightFlow publishes a list of documented Polymarket insider trading cases. Checking whether the model flags those would be a much better test than win rate or edge.
- **The 8% flag rate was picked by hand**, not tuned.

## Running it locally

```
python -m venv polymarket_env
polymarket_env\Scripts\activate
pip install -r src/requirements.txt
```

Create a `.env` file in the repo root with your Turso credentials:

```
TURSO_DATABASE_URL=libsql://...
TURSO_AUTH_TOKEN=...
```

Then from `src/`:

```
python main.py            # score active markets
python backtest.py        # check against resolved markets
python validation.py      # leakage test
streamlit run dashboard.py
```

The same two Turso values also need to be set as GitHub Actions secrets and as Streamlit Cloud secrets.

## Data sources

- [Polymarket Gamma API](https://gamma-api.polymarket.com): market info
- [Polymarket Data API](https://data-api.polymarket.com): trades
