import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("📂 Loading data...")
df = pd.read_csv("charts_combined.csv", parse_dates=["date"])
us = df[(df["region"] == "United States") | (df["region"] == "Global")].copy()
print(f"✅ Rows: {len(us):,}")

print("\n🔧 Engineering fraud signals based on real detection methods...")
daily = us.groupby(["artist", "title", "date"])["streams"].sum().reset_index()
daily = daily.sort_values(["artist", "title", "date"])
daily["month"] = daily["date"].dt.month
daily["year"] = daily["date"].dt.year
daily["day_of_week"] = daily["date"].dt.dayofweek

# ── SIGNAL 1: Stream velocity spike (core signal) ──
daily["prev_streams"] = daily.groupby(["artist", "title"])["streams"].shift(1)
daily["pct_change"] = daily["streams"].div(daily["prev_streams"].replace(0, np.nan)) - 1
daily["rolling_7d_avg"] = daily.groupby(["artist", "title"])["streams"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
daily["spike_ratio"] = daily["streams"] / daily["rolling_7d_avg"].replace(0, np.nan)

# ── SIGNAL 2: Abrupt drop-off after spike (bots stop suddenly) ──
daily["next_streams"] = daily.groupby(["artist", "title"])["streams"].shift(-1)
daily["drop_after_spike"] = (daily["spike_ratio"] > 2) & (
    daily["next_streams"] < daily["rolling_7d_avg"]
)

# ── SIGNAL 3: Seasonal correction ──
monthly_avg = daily.groupby(["artist", "month"])["streams"].transform("mean")
daily["seasonal_spike"] = daily["streams"] / monthly_avg.replace(0, np.nan)

# ── SIGNAL 4: Weekend vs weekday pattern ──
# Real listeners skew toward weekends; bots are flat 24/7
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
weekday_avg = daily.groupby(["artist", "title", "is_weekend"])["streams"].transform("mean")
daily["weekend_ratio"] = daily.groupby(["artist", "title"])["streams"].transform(
    lambda x: x[daily.loc[x.index, "is_weekend"] == 1].mean() /
              (x[daily.loc[x.index, "is_weekend"] == 0].mean() + 1)
)

# ── SIGNAL 5: Stream plateau detection ──
# Bots stream at unnaturally consistent levels (low variance)
daily["rolling_std"] = daily.groupby(["artist", "title"])["streams"].transform(
    lambda x: x.rolling(7, min_periods=3).std()
)
daily["rolling_cv"] = daily["rolling_std"] / daily["rolling_7d_avg"].replace(0, np.nan)

daily = daily.dropna(subset=["pct_change", "spike_ratio", "seasonal_spike", "rolling_cv"])
daily["drop_after_spike"] = daily["drop_after_spike"].astype(float)
print(f"✅ Daily records: {len(daily):,}")

# ── MODEL 1: Isolation Forest ──
print("\n🤖 Running Isolation Forest...")
features = ["streams", "pct_change", "spike_ratio", "seasonal_spike",
            "rolling_cv", "drop_after_spike"]
X = daily[features].replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
daily["iso_score"] = iso.fit_predict(X_scaled)
daily["iso_bot"] = (daily["iso_score"] == -1).astype(int)
print(f"   Isolation Forest flagged: {daily['iso_bot'].sum():,} ({daily['iso_bot'].mean()*100:.1f}%)")

# ── MODEL 2: Local Outlier Factor ──
print("🤖 Running Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
daily["lof_bot"] = (lof.fit_predict(X_scaled) == -1).astype(int)
print(f"   LOF flagged: {daily['lof_bot'].sum():,} ({daily['lof_bot'].mean()*100:.1f}%)")

# ── ENSEMBLE: Only flag if BOTH models agree ──
daily["ensemble_bot"] = ((daily["iso_bot"] == 1) & (daily["lof_bot"] == 1)).astype(int)
daily["bot_stream_val"] = daily["streams"] * daily["ensemble_bot"]
print(f"\n✅ Ensemble (both agree) flagged: {daily['ensemble_bot'].sum():,} ({daily['ensemble_bot'].mean()*100:.1f}%)")

# ── TRACK LEVEL ──
track_stats = daily.groupby(["artist", "title"]).agg(
    total_streams=("streams", "sum"),
    bot_streams=("bot_stream_val", "sum"),
    flagged_days=("ensemble_bot", "sum"),
    total_days=("ensemble_bot", "count"),
    max_spike_ratio=("spike_ratio", "max"),
    avg_rolling_cv=("rolling_cv", "mean"),
    drop_after_spike_count=("drop_after_spike", "sum"),
).reset_index()
track_stats["bot_pct"] = (track_stats["bot_streams"] / track_stats["total_streams"] * 100).round(1)

# ── CONFIDENCE SCORE ──
# Higher confidence when: multiple signals agree, more data points, higher spike ratios
track_stats["confidence"] = (
    (track_stats["flagged_days"] / track_stats["total_days"].clip(1)) * 40 +
    (track_stats["max_spike_ratio"].clip(0, 10) / 10) * 30 +
    (track_stats["drop_after_spike_count"] / track_stats["total_days"].clip(1)) * 30
).clip(0, 100).round(1)

track_stats.to_csv("fraud_results_tracks.csv", index=False)
print(f"✅ Track stats saved — {len(track_stats):,} tracks")

# ── ARTIST LEVEL: split collabs ──
print("\n🔀 Splitting collab tags...")
daily_copy = daily[["artist", "title", "streams", "ensemble_bot", "bot_stream_val", "year", "spike_ratio", "drop_after_spike"]].copy().reset_index(drop=True)
daily_copy["artist_split"] = daily_copy["artist"].str.split(r",\s*|\s*&\s*|\s*ft\.?\s*|\s*feat\.?\s*", regex=True)
daily_copy = daily_copy.explode("artist_split")
daily_copy["artist_split"] = daily_copy["artist_split"].str.strip()
daily_copy = daily_copy[daily_copy["artist_split"].str.len() > 0].reset_index(drop=True)

artist_stats = daily_copy.groupby("artist_split").agg(
    total_streams=("streams", "sum"),
    bot_streams=("bot_stream_val", "sum"),
    flagged_days=("ensemble_bot", "sum"),
    total_days=("ensemble_bot", "count"),
    years_active=("year", "nunique"),
    max_spike_ratio=("spike_ratio", "max"),
    drop_events=("drop_after_spike", "sum"),
).reset_index().rename(columns={"artist_split": "artist"})

artist_stats["bot_pct"] = (artist_stats["bot_streams"] / artist_stats["total_streams"] * 100).round(1)
artist_stats["confidence"] = (
    (artist_stats["flagged_days"] / artist_stats["total_days"].clip(1)) * 40 +
    (artist_stats["max_spike_ratio"].clip(0, 10) / 10) * 30 +
    (artist_stats["drop_events"] / artist_stats["total_days"].clip(1)) * 30
).clip(0, 100).round(1)

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

print("\n📊 Top 20 Most Suspicious Artists (Ensemble Model):")
print(top[["artist", "total_streams", "bot_pct", "confidence", "flagged_days", "drop_events"]].to_string(index=False))

# ── PLOT ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Spotify Bot Stream Detection — Ensemble Model (IF + LOF)", fontsize=14)

colors = ["#e74c3c" if c > 60 else "#f39c12" if c > 30 else "#1DB954"
          for c in top["confidence"][:10]]
axes[0].barh(top["artist"][:10], top["bot_pct"][:10], color=colors)
axes[0].set_title("Bot % — color = confidence (red=high, yellow=med, green=low)")
axes[0].set_xlabel("Bot Stream %")
axes[0].invert_yaxis()

axes[1].scatter(artist_stats["bot_pct"], artist_stats["confidence"],
                alpha=0.5, color="steelblue", s=20)
axes[1].set_title("Bot % vs Confidence Score")
axes[1].set_xlabel("Bot %")
axes[1].set_ylabel("Confidence Score")

plt.tight_layout()
plt.savefig("fraud_analysis.png", dpi=150)
print("\n✅ Saved fraud_results_artists.csv, fraud_results_tracks.csv, fraud_analysis.png")

# ── POST-PROCESSING: Flag known confounders ──
DEATH_SPIKES = {
    "Nipsey Hussle": "2019-03-31",
    "Mac Miller": "2018-09-07",
    "Juice WRLD": "2019-12-08",
    "Pop Smoke": "2020-02-19",
    "Lil Peep": "2017-11-15",
    "XXXTentacion": "2018-06-18",
}

print("\n⚠️  Flagging artists with known death/tribute spikes...")
for artist, date in DEATH_SPIKES.items():
    match = artist_stats[artist_stats["artist"] == artist]
    if len(match) > 0:
        print(f"   {artist} — tribute spike after {date}, reducing confidence")
        artist_stats.loc[artist_stats["artist"] == artist, "confidence"] *= 0.4
        artist_stats.loc[artist_stats["artist"] == artist, "note"] = "⚠️ Death/tribute spike detected — results may be inflated"

artist_stats["note"] = artist_stats.get("note", "")
artist_stats.to_csv("fraud_results_artists.csv", index=False)

top = artist_stats.sort_values("bot_pct", ascending=False).head(20)
print("\n📊 Final Top 20 (post-processed):")
print(top[["artist", "bot_pct", "confidence", "flagged_days"]].to_string(index=False))
