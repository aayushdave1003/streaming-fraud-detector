import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

def get_artist_top_tracks(artist_name):
    print(f"\n🔍 Fetching data for: {artist_name}")

    result = sp.search(q=artist_name, type="artist", limit=1)
    items = result["artists"]["items"]
    if not items:
        print(f"   ❌ Artist not found")
        return []

    artist = items[0]
    artist_id = artist["id"]
    print(f"   Artist: {artist['name']} (id: {artist_id})")

    top_tracks = sp.artist_top_tracks(artist_id, country="US")["tracks"]
    print(f"   Found {len(top_tracks)} top tracks")

    tracks = []
    for track in top_tracks:
        try:
            features = sp.audio_features(track["id"])[0]
            tracks.append({
                "artist": artist["name"],
                "artist_id": artist_id,
                "track_name": track["name"],
                "track_id": track["id"],
                "duration_ms": track["duration_ms"],
                "explicit": track["explicit"],
                "danceability": features["danceability"] if features else None,
                "energy": features["energy"] if features else None,
                "valence": features["valence"] if features else None,
                "tempo": features["tempo"] if features else None,
                "acousticness": features["acousticness"] if features else None,
                "instrumentalness": features["instrumentalness"] if features else None,
                "speechiness": features["speechiness"] if features else None,
            })
        except Exception as e:
            print(f"   ⚠️ Skipped track {track.get('name', '?')}: {e}")

    return tracks

artists = [
    "Drake",
    "Taylor Swift",
    "Bad Bunny",
    "The Weeknd",
    "Peso Pluma"
]

all_tracks = []
for artist in artists:
    try:
        tracks = get_artist_top_tracks(artist)
        all_tracks.extend(tracks)
    except Exception as e:
        print(f"❌ Error fetching {artist}: {e}")

if all_tracks:
    df = pd.DataFrame(all_tracks)
    df.to_csv("spotify_real_data.csv", index=False)
    print(f"\n✅ Done! {len(df)} tracks saved to spotify_real_data.csv")
    print(df[["artist", "track_name", "duration_ms", "energy", "danceability"]].to_string(index=False))
else:
    print("\n❌ No data fetched")
