import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stream Fraud Detector", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e0e0e; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1DB954; }
</style>
""", unsafe_allow_html=True)

SPOTIFY_RATE = 0.004  # avg $ per stream


@st.cache_data
def _load_charts():
    """Load chart history, preferring the fast curated parquet.

    Degrades gracefully: if neither the parquet nor the raw CSV is present
    (e.g. a fresh clone without the large gitignored data), the dashboard
    still runs — the per-artist timelines just show "no data".
    """
    for path in ("charts_us_global.parquet", "charts_combined.csv"):
        if os.path.exists(path):
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=["artist", "title", "date", "region", "streams"])


@st.cache_data
def load_data():
    artists = pd.read_csv("fraud_results_artists.csv")
    tracks = pd.read_csv("fraud_results_tracks.csv")
    genres = pd.read_csv("artist_genres.csv")
    rw = pd.read_csv("real_world_reports.csv")
    charts = _load_charts()
    artists = artists.merge(genres[["artist", "genre_category"]], on="artist", how="left")
    artists["genre_category"] = artists["genre_category"].fillna("Other")
    artists["royalty_lost"] = (artists["bot_streams"] * SPOTIFY_RATE).round(0)
    tracks["royalty_lost"] = (tracks["bot_streams"] * SPOTIFY_RATE).round(0)
    return artists, tracks, genres, rw, charts


def mask_low_confidence(df, reveal_conf, on, name_col="artist"):
    """Name-surfacing guardrail: hide identities of low-confidence flags.

    Anomaly ≠ proof of fraud, so unproven low-confidence artists should not be
    publicly named. Below ``reveal_conf`` the name is replaced with an
    anonymous placeholder.
    """
    if not on or "confidence" not in df.columns:
        return df
    df = df.copy()
    masked = df["confidence"] < reveal_conf
    df.loc[masked, name_col] = [f"🔒 Under review #{i + 1}" for i in range(int(masked.sum()))]
    return df

artists_df, tracks_df, genres_df, rw_df, charts_df = load_data()

# ── HEADER ──
st.title("🎵 Spotify Stream Fraud Detector")
st.markdown("Detecting bot-driven stream manipulation using anomaly detection on 26M+ real Spotify chart entries.")
st.divider()

# ── SIDEBAR ──
st.sidebar.title("🎛️ Filters")
genres = ["All Genres"] + sorted(artists_df["genre_category"].dropna().unique().tolist())
selected_genre = st.sidebar.selectbox("Genre", genres)
min_bot_pct = st.sidebar.slider("Min Bot %", 0, 100, 0)
min_streams = st.sidebar.number_input("Min Total Streams", value=0, step=1_000_000)

st.sidebar.divider()
st.sidebar.subheader("🔍 Confidence & safeguards")
min_conf = st.sidebar.slider("Min confidence", 0, 100, 0,
                             help="Hide flags the model is not confident about.")
guardrail_on = st.sidebar.checkbox(
    "🔒 Mask names below a confidence threshold", value=True,
    help="Ethics guardrail: anomaly is not proof of fraud, so low-confidence "
         "artists are shown anonymously rather than publicly named.")
reveal_conf = st.sidebar.slider("Reveal names at confidence ≥", 0, 100, 0,
                                disabled=not guardrail_on)

filtered = artists_df.copy()
if selected_genre != "All Genres":
    filtered = filtered[filtered["genre_category"] == selected_genre]
filtered = filtered[filtered["bot_pct"] >= min_bot_pct]
filtered = filtered[filtered["total_streams"] >= min_streams]
if "confidence" in filtered.columns:
    filtered = filtered[filtered["confidence"] >= min_conf]
filtered = filtered.sort_values("bot_pct", ascending=False)

# ── TOP METRICS ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎤 Artists Analyzed", f"{len(filtered):,}")
col2.metric("🤖 Avg Bot %", f"{filtered['bot_pct'].mean():.1f}%")
col3.metric("💰 Est. Royalties Lost", f"${filtered['royalty_lost'].sum()/1e6:.1f}M")
col4.metric("⚠️ High Risk Artists", f"{len(filtered[filtered['bot_pct'] > 10]):,}")

st.divider()

# ── GENRE TABS ──
st.subheader("📊 Browse by Genre")
genre_list = ["All", "Hip Hop / Rap", "R&B / Soul", "Pop", "Latin", "Country", "EDM / Electronic", "Rock / Alternative", "Other"]
tabs = st.tabs(genre_list)

for tab, genre in zip(tabs, genre_list):
    with tab:
        tab_df = artists_df.copy() if genre == "All" else artists_df[artists_df["genre_category"] == genre].copy()
        tab_df = tab_df.sort_values("bot_pct", ascending=False)

        if len(tab_df) == 0:
            st.info("No artists found.")
            continue

        top10 = tab_df.head(10)
        fig = px.bar(
            top10, x="bot_pct", y="artist",
            orientation="h",
            color="bot_pct",
            color_continuous_scale=["#1DB954", "#f39c12", "#e74c3c"],
            labels={"bot_pct": "Bot Stream %", "artist": "Artist"},
            title=f"Most Suspicious Artists — {genre}"
        )
        fig.update_layout(
            paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
            font_color="white", yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, height=400
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{genre}")

        # ── ARTIST DRILL DOWN ──
        st.markdown("#### 🔍 Select an artist to investigate")
        selected_artist = st.selectbox("Artist", tab_df["artist"].tolist(), key=f"select_{genre}")

        if selected_artist:
            row = tab_df[tab_df["artist"] == selected_artist].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Streams", f"{row['total_streams']/1e6:.1f}M")
            c2.metric("Bot Streams", f"{row['bot_streams']/1e6:.1f}M")
            c3.metric("Bot %", f"{row['bot_pct']}%")
            c4.metric("💰 Royalties Lost", f"${row['royalty_lost']:,.0f}")

            # ── STREAM TIMELINE ──
            st.markdown(f"#### 📈 Stream Timeline — {selected_artist}")
            artist_charts = charts_df[
                charts_df["artist"].str.contains(selected_artist, case=False, na=False)
            ].groupby("date")["streams"].sum().reset_index()

            if len(artist_charts) > 0:
                # Get flagged dates
                artist_tracks = tracks_df[
                    tracks_df["artist"].str.contains(selected_artist, case=False, na=False)
                ]

                fig_timeline = go.Figure()

                # Main stream line
                fig_timeline.add_trace(go.Scatter(
                    x=artist_charts["date"],
                    y=artist_charts["streams"],
                    mode="lines",
                    name="Daily Streams",
                    line=dict(color="#1DB954", width=2)
                ))

                # 7-day rolling average
                artist_charts["rolling_avg"] = artist_charts["streams"].rolling(7, min_periods=1).mean()
                fig_timeline.add_trace(go.Scatter(
                    x=artist_charts["date"],
                    y=artist_charts["rolling_avg"],
                    mode="lines",
                    name="7-Day Avg",
                    line=dict(color="#aaaaaa", width=1, dash="dash")
                ))

                # Spike threshold line
                spike_threshold = artist_charts["rolling_avg"] * 2
                fig_timeline.add_trace(go.Scatter(
                    x=artist_charts["date"],
                    y=spike_threshold,
                    mode="lines",
                    name="Spike Threshold (2x avg)",
                    line=dict(color="#e74c3c", width=1, dash="dot"),
                    opacity=0.5
                ))

                # Highlight spikes in red
                spikes = artist_charts[artist_charts["streams"] > spike_threshold]
                if len(spikes) > 0:
                    fig_timeline.add_trace(go.Scatter(
                        x=spikes["date"],
                        y=spikes["streams"],
                        mode="markers",
                        name="🚨 Suspicious Spike",
                        marker=dict(color="#e74c3c", size=8, symbol="circle")
                    ))

                fig_timeline.update_layout(
                    paper_bgcolor="#111", plot_bgcolor="#111",
                    font_color="white",
                    xaxis_title="Date",
                    yaxis_title="Streams",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_timeline, use_container_width=True, key=f"timeline_{genre}_{selected_artist}")
            else:
                st.info("No timeline data available for this artist.")

            # ── TRACK BREAKDOWN ──
            artist_tracks = tracks_df[
                tracks_df["artist"].str.contains(selected_artist, case=False, na=False)
            ].sort_values("bot_pct", ascending=False)

            if len(artist_tracks) > 0:
                st.markdown(f"#### 🎵 Track Breakdown — {selected_artist}")

                fig_tracks = px.bar(
                    artist_tracks.head(15),
                    x="bot_pct", y="title",
                    orientation="h",
                    color="bot_pct",
                    color_continuous_scale=["#1DB954", "#f39c12", "#e74c3c"],
                    labels={"bot_pct": "Bot %", "title": "Track"},
                )
                fig_tracks.update_layout(
                    paper_bgcolor="#111", plot_bgcolor="#111",
                    font_color="white", yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False, height=400
                )
                st.plotly_chart(fig_tracks, use_container_width=True, key=f"tracks_{genre}_{selected_artist}")

                # Royalty table
                st.dataframe(
                    artist_tracks[["title", "total_streams", "bot_streams", "bot_pct", "royalty_lost", "flagged_days"]].rename(columns={
                        "title": "Track",
                        "total_streams": "Total Streams",
                        "bot_streams": "Bot Streams",
                        "bot_pct": "Bot %",
                        "royalty_lost": "💰 Royalties Lost ($)",
                        "flagged_days": "Flagged Days"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

st.divider()

# ── ROYALTY IMPACT ──
st.subheader("💰 Royalty Impact Calculator")
st.caption(f"Based on Spotify's average payout of ${SPOTIFY_RATE}/stream")

col1, col2 = st.columns(2)
with col1:
    total_bot = filtered["bot_streams"].sum()
    total_lost = filtered["royalty_lost"].sum()
    st.metric("Total Suspected Bot Streams", f"{total_bot/1e9:.2f}B")
    st.metric("Total Estimated Royalties Lost", f"${total_lost/1e6:.1f}M")
    st.metric("Avg Lost Per Artist", f"${filtered['royalty_lost'].mean():,.0f}")

with col2:
    top_royalty = mask_low_confidence(
        filtered.nlargest(10, "royalty_lost"), reveal_conf, guardrail_on
    )[["artist", "bot_streams", "royalty_lost"]]
    top_royalty["bot_streams"] = (top_royalty["bot_streams"]/1e6).round(1).astype(str) + "M"
    top_royalty["royalty_lost"] = top_royalty["royalty_lost"].apply(lambda x: f"${x:,.0f}")
    st.markdown("**Top 10 Artists by Estimated Royalties Lost**")
    st.dataframe(top_royalty.rename(columns={
        "artist": "Artist",
        "bot_streams": "Bot Streams",
        "royalty_lost": "Royalties Lost"
    }), hide_index=True, use_container_width=True)

st.divider()

# ── REAL WORLD REPORTS ──
st.subheader("🌍 Real World Verified Incidents")
st.caption("Stream removals confirmed by credible industry sources.")

verified = rw_df[rw_df["verified"] == True]
total_removed = verified["streams_removed"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("📋 Incidents Documented", len(rw_df))
c2.metric("✅ Verified Incidents", len(verified))
c3.metric("🗑️ Confirmed Streams Removed", f"{total_removed/1e6:.0f}M")

st.divider()
for _, row in rw_df.iterrows():
    verified_badge = "✅ Verified" if row["verified"] else "⚠️ Alleged"
    legal_badge = " ⚖️ Legal Action" if row["legal_action"] else ""
    removed = f"{row['streams_removed']/1e6:.0f}M streams removed" if pd.notna(row["streams_removed"]) else "Stream count unconfirmed"
    with st.expander(f"{verified_badge}{legal_badge} — **{row['artist']}** ({row['year']}) — {removed}"):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Label:** {row['label']}")
        c2.markdown(f"**Source:** {row['source']}")
        st.markdown(f"**What happened:** {row['incident']}")

st.divider()

# ── FULL LEADERBOARD ──
st.subheader("🏆 Full Artist Leaderboard")
if guardrail_on:
    st.caption(f"🔒 Guardrail on — artists below confidence {reveal_conf} are shown anonymously.")

lb_cols = ["artist", "genre_category", "total_streams", "bot_streams", "bot_pct",
           "royalty_lost", "confidence", "flagged_days"]
if "flag_reasons" in filtered.columns:      # explainability: why each artist was flagged
    lb_cols.append("flag_reasons")
if "note" in filtered.columns:              # confounder annotations (holiday / death spike)
    lb_cols.append("note")
leaderboard = mask_low_confidence(filtered, reveal_conf, guardrail_on)
display_df = leaderboard[lb_cols].rename(columns={
    "artist": "Artist",
    "genre_category": "Genre",
    "total_streams": "Total Streams",
    "bot_streams": "Bot Streams",
    "bot_pct": "Bot %",
    "royalty_lost": "💰 Royalties Lost ($)",
    "confidence": "Confidence",
    "flagged_days": "Flagged Days",
    "flag_reasons": "Why flagged",
    "note": "Note",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
