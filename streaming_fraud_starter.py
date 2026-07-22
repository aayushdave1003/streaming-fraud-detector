import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_sessions = 10000

df = pd.DataFrame({
    "session_id": range(n_sessions),
    "artist": np.random.choice(["Artist_A", "Artist_B", "Artist_C", "Artist_D"], n_sessions),
    "track": np.random.choice([f"Track_{i}" for i in range(50)], n_sessions),
    "listen_duration_sec": np.random.normal(180, 60, n_sessions).clip(10, 400),
    "track_duration_sec": np.random.normal(200, 40, n_sessions).clip(60, 400),
    "skipped": np.random.choice([0, 1], n_sessions, p=[0.4, 0.6]),
    "repeat_count": np.random.exponential(1.5, n_sessions).astype(int).clip(0, 50),
    "hour_of_day": np.random.randint(0, 24, n_sessions),
    "session_length_tracks": np.random.exponential(5, n_sessions).astype(int).clip(1, 100),
    "unique_tracks_in_session": np.random.exponential(4, n_sessions).astype(int).clip(1, 100),
})

bot_idx = df.index[-500:]
df.loc[bot_idx, "skipped"] = 0
df.loc[bot_idx, "repeat_count"] = np.random.randint(20, 50, 500)
df.loc[bot_idx, "listen_duration_sec"] = df.loc[bot_idx, "track_duration_sec"]
df.loc[bot_idx, "hour_of_day"] = np.random.choice([2, 3, 4], 500)
df.loc[bot_idx, "unique_tracks_in_session"] = 1

print("✅ Data loaded")
print(df.shape)

df["completion_rate"] = (df["listen_duration_sec"] / df["track_duration_sec"]).clip(0, 1)
df["session_diversity"] = df["unique_tracks_in_session"] / df["session_length_tracks"].clip(1)
df["odd_hour"] = df["hour_of_day"].apply(lambda x: 1 if x in range(2, 6) else 0)
df["high_repeat"] = (df["repeat_count"] > 10).astype(int)

print("✅ Features engineered")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Streaming Session Behavior Analysis", fontsize=16)
axes[0, 0].hist(df["completion_rate"], bins=50, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Completion Rate Distribution")
axes[0, 1].hist(df["repeat_count"].clip(0, 30), bins=30, color="coral", edgecolor="white")
axes[0, 1].set_title("Repeat Count Distribution")
hour_counts = df.groupby("hour_of_day").size()
axes[1, 0].bar(hour_counts.index, hour_counts.values, color="mediumseagreen")
axes[1, 0].set_title("Streams by Hour of Day")
axes[1, 1].hist(df["session_diversity"], bins=50, color="mediumpurple", edgecolor="white")
axes[1, 1].set_title("Session Diversity")
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
print("✅ EDA plots saved")

features = ["completion_rate", "session_diversity", "repeat_count", "skipped", "odd_hour", "high_repeat"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)
df["anomaly_score"] = model.decision_function(X_scaled)
df["is_bot"] = (model.predict(X_scaled) == -1).astype(int)

print(f"✅ Model trained — Flagged as bot: {df['is_bot'].sum()} ({df['is_bot'].mean()*100:.1f}%)")

artist_stats = df.groupby("artist").agg(
    total_streams=("session_id", "count"),
    bot_streams=("is_bot", "sum"),
).reset_index()
artist_stats["real_streams"] = artist_stats["total_streams"] - artist_stats["bot_streams"]
artist_stats["bot_percentage"] = (artist_stats["bot_streams"] / artist_stats["total_streams"] * 100).round(1)

print("\n📊 Bot % Per Artist:")
print(artist_stats.sort_values("bot_percentage", ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["crimson" if p > 10 else "steelblue" for p in artist_stats["bot_percentage"]]
bars = ax.bar(artist_stats["artist"], artist_stats["bot_percentage"], color=colors)
ax.axhline(y=10, color="orange", linestyle="--", label="10% threshold")
ax.set_title("Estimated Bot Stream % Per Artist", fontsize=14)
ax.set_ylabel("Bot Percentage (%)")
for bar, val in zip(bars, artist_stats["bot_percentage"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val}%", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("bot_percentage_per_artist.png", dpi=150)
print("✅ Bot % chart saved")
