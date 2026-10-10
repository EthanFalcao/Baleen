import requests
import json
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass  # not installed in prod (GitHub Actions/Streamlit Cloud set real env vars directly)

GAMMA_API  = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"


def _get_secret(key):
    """Reads a secret from the environment first (GitHub Actions secrets,
    local .env via python-dotenv above). Falls back to Streamlit's
    st.secrets when running on Streamlit Cloud, since that's the only
    consumer of this module that might have secrets configured there instead
    of as plain env vars."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        raise RuntimeError(
            f"Missing required secret '{key}'. Set it as an env var (GitHub Actions "
            f"secret, local .env) or in Streamlit Cloud's app secrets."
        )


# DB now lives in Turso (SQLite-compatible, hosted) instead of a file
# committed to git — the committed src/polymarket.db kept growing past
# GitHub's 100MB file size limit and started failing the daily pipeline push.
TURSO_DATABASE_URL = _get_secret("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = _get_secret("TURSO_AUTH_TOKEN")

HEADERS    = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TOP_N_MARKETS     = 2000
TRADES_PER_MARKET = 1000

print("Ready.")