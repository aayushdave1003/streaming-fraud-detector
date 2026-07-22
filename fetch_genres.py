import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

df = pd.read_csv("fraud_results_artists.csv")
artists = df["artist"].unique().tolist()
print(f"🎵 Fetching genres for {len(artists)} artists...")

results = []
for i, name in enumerate(artists):
    try:
        res = sp.search(q=name, type="artist", limit=1)
        items = res["artists"]["items"]
        if items:
            genres = sp.artist(items[0]["id"]).get("genres", [])
            # Map raw genres to broad categories
            genre_str = ", ".join(genres)
            if any(g in genre_str for g in ["hip hop", "rap", "trap"]):
                broad = "Hip Hop / Rap"
            elif any(g in genre_str for g in ["pop", "dance pop", "electropop"]):
                broad = "Pop"
            elif any(g in genre_str for g in ["r&b", "soul", "urban"]):
                broad = "R&B / Soul"
            elif any(g in genre_str for g in ["rock", "indie", "alternative"]):
                broad = "Rock / Alternative"
            elif any(g in genre_str for g in ["latin", "reggaeton", "urbano"]):
                broad = "Latin"
            elif any(g in genre_str for g in ["country"]):
                broad = "Country"
            elif any(g in genre_str for g in ["edm", "electronic", "house", "techno"]):
                broad = "EDM / Electronic"
            else:
                broad = "Other"
            results.append({"artist": name, "genres": genre_str, "genre_category": broad})
            print(f"  [{i+1}/{len(artists)}] {name}: {broad} ({genre_str[:50]})")
        else:
            results.append({"artist": name, "genres": "", "genre_category": "Other"})
        time.sleep(0.2)
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        results.append({"artist": name, "genres": "", "genre_category": "Other"})

genre_df = pd.DataFrame(results)
genre_df.to_csv("artist_genres.csv", index=False)
print(f"\n✅ Saved artist_genres.csv with {len(genre_df)} artists")
