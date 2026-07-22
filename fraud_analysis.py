import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("📂 Loading data...")
df = pd.read_csv("charts_combined.csv", parse_dates=["date"])
us = df[(df["region"] == "United States") | (df["region"] == "Global")].copy()
print(f"✅ US chart rows: {len(us):,}")

print("\n🔧 Engineering fraud signals...")
daily = us.groupby(["artist", "title", "date"])["streams"].sum().reset_index()
daily = daily.sort_values(["artist", "title", "date"])
daily["month"] = daily["date"].dt.month
daily["year"] = daily["date"].dt.year
daily["prev_streams"] = daily.groupby(["artist", "title"])["streams"].shift(1)
daily["pct_change"] = daily["streams"].div(daily["prev_streams"].replace(0, np.nan)) - 1
daily["rolling_7d_avg"] = daily.groupby(["artist", "title"])["streams"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
daily["spike_ratio"] = daily["streams"] / daily["rolling_7d_avg"].replace(0, np.nan)
monthly_avg = daily.groupby(["artist", "month"])["streams"].transform("mean")
daily["seasonal_spike"] = daily["streams"] / monthly_avg.replace(0, np.nan)
daily = daily.dropna()
print(f"✅ Daily records: {len(daily):,}")

print("\n🤖 Training anomaly detector...")
features = ["streams", "pct_change", "spike_ratio", "seasonal_spike"]
X = daily[features].replace([np.inf, -np.inf], np.nan).dropna()
valid_idx = X.index
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
model.fit(X_scaled)
daily.loc[valid_idx, "anomaly_score"] = model.decision_function(X_scaled)
daily.loc[valid_idx, "is_bot"] = (model.predict(X_scaled) == -1).astype(int)
daily["bot_stream_val"] = daily["streams"] * daily["is_bot"]
print(f"✅ Flagged {daily['is_bot'].sum():,.0f} suspicious spikes")

# ── TRACK-LEVEL ──
track_stats = daily.groupby(["artist", "title"]).agg(
    total_streams=("streams", "sum"),
    bot_streams=("bot_stream_val", "sum"),
    flagged_days=("is_bot", "sum"),
    total_days=("is_bot", "count"),
).reset_index()
track_stats["bot_pct"] = (track_stats["bot_streams"] / track_stats["total_streams"] * 100).round(1)
track_stats.to_csv("fraud_results_tracks.csv", index=False)
print(f"✅ Track-level saved — {len(track_stats):,} tracks")

# ── ARTIST-LEVEL: split collabs ──
print("\n🔀 Splitting collab artist tags...")
daily_copy = daily[["artist", "title", "streams", "is_bot", "bot_stream_val", "year"]].copy()
daily_copy = daily_copy.reset_index(drop=True)

# Split on comma, &, ft., feat.
daily_copy["artist_split"] = daily_copy["artist"].str.split(r",\s*|\s*&\s*|\s*ft\.?\s*|\s*feat\.?\s*", regex=True)
daily_copy = daily_copy.explode("artist_split")
daily_copy["artist_split"] = daily_copy["artist_split"].str.strip()
daily_copy = daily_copy[daily_copy["artist_split"].str.len() > 0].reset_index(drop=True)

# Aggregate — no lambda needed
artist_stats = daily_copy.groupby("artist_split").agg(
    total_streams=("streams", "sum"),
    bot_streams=("bot_stream_val", "sum"),
    flagged_days=("is_bot", "sum"),
    total_days=("is_bot", "count"),
    years_active=("year", "nunique"),
).reset_index().rename(columns={"artist_split": "artist"})

artist_stats["bot_pct"] = (artist_stats["bot_streams"] / artist_stats["total_streams"] * 100).round(1)

HOLIDAY = ["Brenda Lee", "Bing Crosby", "Bobby Helms", "Perry Como",
           "Nat King Cole", "Andy Williams", "Burl Ives", "Dean Martin",
           "Mariah Carey", "Vince Guaraldi Trio", "Frank Sinatra",
           "Darlene Love", "The Ronettes", "José Feliciano", "Gene Autry",
           "Carpenters", "Ella Fitzgerald", "Tony Bennett", "Chris Rea"]

artist_stats = artist_stats[
    (artist_stats["total_days"] >= 90) &
    (artist_stats["total_streams"] >= 5_000_000) &
    (artist_stats["years_active"] >= 2) &
    (~artist_stats["artist"].isin(HOLIDAY))
]

artist_stats.to_csv("fraud_results_artists.csv", index=False)
top = artist_stats.sort_values("bot_pct", ascending=False).head(20)

print("\n📊 Top 20 Most Suspicious Artists:")
print(top[["artist", "total_streams", "bot_pct", "flagged_days", "total_days"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(top["artist"][:10], top["bot_pct"][:10], color="crimson")
ax.set_title("Top 10 Most Suspicious Artists — Bot Stream %", fontsize=14)
ax.set_xlabel("Suspicious Stream %")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("fraud_analysis.png", dpi=150)
print("\n✅ Saved fraud_results_artists.csv, fraud_results_tracks.csv, fraud_analysis.png")
