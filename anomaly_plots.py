#!/usr/bin/env python
# Stage B: read the master CSV, detect anomalies, create plots, print summary

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Config for anomaly logic ───────────────────
Z_THRESH = 3.0
WINDOW   = 10
MIN_HITS = 3
VOL_GAP  = 0.4             # fund σ < 40 % of peer median ⇒ “smooth” alert
# ────────────────────────────────────────────────

# 1  Load master table (Ticker, Date, NAV, Return)
df_all = pd.read_csv("all_funds_nav.csv", parse_dates=["Date"])
tickers = df_all["Ticker"].unique()

# 2  Tag outliers & burst clusters per ticker
def tag(df):
    z = (df["Return"] - df["Return"].mean()) / df["Return"].std()
    df["Outlier"] = z.abs() > Z_THRESH
    df["Cluster"] = df["Outlier"] & (df["Outlier"].rolling(WINDOW, 1).sum() >= MIN_HITS)
    return df

df_tagged = pd.concat([tag(g.copy()) for _, g in df_all.groupby("Ticker")],
                      ignore_index=True)

# 3  60‑day rolling volatility per ticker
def rolling_vol(sub):
    return (sub.set_index("Date")["Return"]
              .rolling(60).std()
              .rename(sub["Ticker"].iloc[0]))

vol_df = pd.concat([rolling_vol(g) for _, g in df_tagged.groupby("Ticker")],
                   axis=1)  # Date index × ticker columns

# 4  Combined volatility chart
plt.figure(figsize=(10,5))
for tkr in tickers:
    plt.plot(vol_df.index, vol_df[tkr], lw=1, label=tkr)
peer_med = vol_df.median(axis=1)
plt.plot(peer_med.index, peer_med, lw=2.5, color="black", label="Peer median σ")
plt.title("60‑day Rolling Volatility – All Tickers")
plt.ylabel("σ (std of daily return)")
plt.grid(alpha=.3); plt.legend(ncol=3, fontsize=8)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout(); plt.savefig("all_vol_plot.png", dpi=150); plt.close()

# 5  Per‑fund burst CSVs, NAV plots, and classification summary
for tkr in tickers:
    sub = df_tagged[df_tagged["Ticker"] == tkr].copy()

    # — save burst rows —
    sub[sub["Cluster"]].to_csv(f"{tkr.lower()}_bursts.csv", index=False)

    # — NAV + outliers/bursts plot —
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(sub["Date"], sub["NAV"], lw=1.2)
    ax1.scatter(sub.loc[sub["Outlier"], "Date"],
                sub.loc[sub["Outlier"], "NAV"], color="red", s=18, label="Outlier")
    ax1.scatter(sub.loc[sub["Cluster"], "Date"],
                sub.loc[sub["Cluster"], "NAV"], color="purple", s=26, label="Burst")
    ax1.set_title(f"{tkr} NAV"); ax1.set_ylabel("Price"); ax1.grid(alpha=.3); ax1.legend(fontsize=8)

    ax2.plot(sub["Date"], sub["Return"]*100, lw=.8)
    ax2.axhline(0, color="black", lw=.7)
    ax2.set_ylabel("Return (%)"); ax2.grid(alpha=.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(f"{tkr.lower()}_nav_plot.png", dpi=150); plt.close()

    # — classification summary —
    burst_days = int(sub["Cluster"].sum())
    outliers   = int(sub["Outlier"].sum())

    fund_vol  = vol_df[tkr]
    peer_med  = vol_df.drop(columns=tkr).median(axis=1)
    aligned   = sub.set_index("Date").reindex(fund_vol.index)
    # smooth‑alert count = when fund volatility is much lower than peers, but return is positive
    smooth_mask = ((fund_vol / peer_med) < VOL_GAP) & (aligned["Return"] > 0)
    smooth_ct = int(smooth_mask.sum())

    if burst_days >= 10 and smooth_ct >= 50:
        label = "Anomaly"
    elif burst_days >= 10:
        label = "Burst"
    elif outliers > 0:
        label = "Normal (outliers only)"
    else:
        label = "No anomaly"

    print(f"{tkr}: rows {len(sub):,} | bursts {burst_days} | "
          f"outliers {outliers} | smooth {smooth_ct} | {label}")

print("\nStage B complete – burst CSVs, plots, and summaries created.")
