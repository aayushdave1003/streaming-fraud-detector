import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

result = sp.search(q="Drake", type="artist", limit=1)
artist_id = result["artists"]["items"][0]["id"]

# fetch full artist object
artist = sp.artist(artist_id)
print(artist)
