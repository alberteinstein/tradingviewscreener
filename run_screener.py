#!/usr/bin/env python3
# run_screener.py

from tradingview_screener import Query, Column
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───────────────────────────────────────────────────────────────

# Local timezone for timestamping
LOCAL_TZ = pytz.timezone('Europe/Berlin')  # adjust if needed

# QuickFS API fields and headers
_QFS_FIELDS = [
    "qfs_symbol", "name", "symbol", "exchange", "country", "currency",
    "sector", "industry", "price", "mkt_cap", "pe", "pb", "ps",
    "ev_ebitda", "beta", "avg_vol_50d", "dividend_date",
    "ex_dividend_date", "description"
]

_QFS_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0"
}

# ─── Screener Query ─────────────────────────────────────────────────────────────

def fetch_tradingview():
    """Run TradingView screener and return a cleaned DataFrame."""
    result = (
        Query()
        .select(
            'exchange', 'name', 'description', 'sector', 'industry', 'indexes',
            'type', 'market_cap_basic', 'High.1M', 'Low.1M',
            'price_52_week_high', 'price_52_week_low', 'relative_volume_10d_calc',
            'earnings_release_date', 'earnings_release_next_date',
            'eps_surprise_fq', 'revenue_surprise_percent_fq',
            'recommendation_total', 'recommendation_buy',
            'recommendation_mark', 'price_target_1y', 'typespecs',
            'Perf.1M', 'Perf.3M', 'Perf.6M', 'Perf.Y', 'Perf.W'
        )
        .where(
            Column('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
            Column('type') == 'stock',
            Column('typespecs').has_none_of('preferred'),
        )
        .limit(10000)
        .order_by('name', ascending=True, nulls_first=False)
        .get_scanner_data()
    )

    df = result[1]

    # Normalize list/dict cells to strings
    df = df.applymap(lambda x: str(x) if isinstance(x, (list, dict)) else x)

    # Convert epoch timestamps to readable dates
    df['earnings_release_date'] = (
        pd.to_datetime(df['earnings_release_date'], unit='s')
          .dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    df['earnings_release_next_date'] = (
        pd.to_datetime(df['earnings_release_next_date'], unit='s')
          .dt.strftime('%Y-%m-%d %H:%M:%S')
    )

    # Clean infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna('')

    # Extract index names from nested structures
    def extract_index_names(val):
        if isinstance(val, list):
            return ', '.join(item.get('name', '') for item in val)
        if isinstance(val, dict):
            return val.get('name', '')
        try:
            # maybe a stringified list/dict
            parsed = pd.io.json.loads(val)
            if isinstance(parsed, list):
                return ', '.join(i.get('name', '') for i in parsed)
            if isinstance(parsed, dict):
                return parsed.get('name', '')
        except Exception:
            pass
        return ''

    df['indexes'] = df['indexes'].apply(extract_index_names)

    # Add a retrieval timestamp
    now = datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')
    df.insert(0, 'retrieval_time', now)

    return df

# ─── QuickFS Metadata Fetch ────────────────────────────────────────────────────

def fetch_all_with_us(df, symbol_col="symbol", country_col=None, max_workers=30):
    """
    For each ticker in df[symbol_col], fetch metadata from QuickFS and
    return a combined DataFrame prefixed with 'qfs_'.
    """
    # build list of (symbol, country) pairs
    if country_col and country_col in df.columns:
        pairs = list(df[[symbol_col, country_col]].itertuples(index=False, name=None))
    else:
        pairs = [(sym, "US") for sym in df[symbol_col]]

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def _fetch_meta(sym, ctry):
        tk = f"{sym}:{ctry}"
        url = f"https://api.quickfs.net/stocks/{tk}/ovr/Quarter/"
        try:
            r = session.get(url, headers=_QFS_HEADERS, timeout=10)
            r.raise_for_status()
            meta = r.json()["datasets"]["metadata"]
            out = {f: meta.get(f) for f in _QFS_FIELDS}
        except Exception as e:
            print(f"⚠️ Failed {tk}: {e}")
            out = {f: None for f in _QFS_FIELDS}
        out["qfs_ticker"] = tk
        return out

    results = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_meta, sym, ctry): idx
            for idx, (sym, ctry) in enumerate(pairs)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    meta_df = pd.DataFrame(results).rename(columns=lambda c: f"qfs_{c}")
    # Convert date fields
    for col in ("qfs_dividend_date", "qfs_ex_dividend_date"):
        if col in meta_df:
            meta_df[col] = (
                pd.to_datetime(meta_df[col], format="%Y%m%d", errors="coerce")
                  .dt.strftime("%Y-%m-%d")
            )

    return pd.concat([df.reset_index(drop=True), meta_df], axis=1)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1) Run the TradingView screener
    df = fetch_tradingview()

    # 2) Fetch QuickFS metadata by ticker name
    df2 = fetch_all_with_us(df, symbol_col="name")

    # 3) Final clean-up: replace any remaining infs/nans
    df2 = df2.replace([np.inf, -np.inf], np.nan).fillna('')

    # 4) Output: here we just print the first few rows
    print(df2.head().to_string(index=False))

if __name__ == "__main__":
    main()
