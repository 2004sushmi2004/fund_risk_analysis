#!/usr/bin/env python
# Stage A: download NAVs and write all_funds_nav.csv

import yfinance as yf
import pandas as pd

TICKERS = ["IQDNX", "DSU", "FSCO", "PDO", "KIO","FRA"]
START   = "2014-01-01"
END     = None  # today

def download_nav(tkr):
    """
    Return a tidy DataFrame with 4 plain columns:
    Ticker | Date | NAV | Return
    """
    # grab the Close series only → 1‑D, no MultiIndex
    raw = yf.download(tkr, start=START, end=END,
                  auto_adjust=True, progress=False)

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):    # happens when multi-index returned
        close = close[tkr]
    close = close.dropna()


    df = close.to_frame(name="NAV").reset_index()      # Date | NAV
    df["Return"] = df["NAV"].pct_change()
    df.insert(0, "Ticker", tkr)                        # add first col
    return df[["Ticker", "Date", "NAV", "Return"]]


# Download fresh data
frames = []
for tkr in TICKERS:
    df = download_nav(tkr)
    print(f"{tkr}: {len(df)} rows, columns: {df.columns.tolist()}")
    frames.append(df)

# Combine and write
combined = pd.concat(frames, ignore_index=True)
print("Final columns:", combined.columns.tolist())
combined.to_csv("all_funds_nav.csv", index=False)
print(f"all_funds_nav.csv written with {len(combined):,} rows.")
